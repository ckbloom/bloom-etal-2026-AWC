"""
zhang_and_schaap.py
====================
MvG parameter computation using the pedotransfer function of:
    Zhang, Y., Schaap, M.G. (2017) — Rosetta3, via the `rosetta-soil` package.

Approach
--------
No closed-form equations are evaluated; θs, θr, α, and n are predicted by
the Rosetta3 neural network ensemble (model_code=3: sand/silt/clay + bulk
density), run through the `Rosetta` class from `rosetta-soil`.

where:
    sand : sand content     (%)
    silt : silt content     (%)
    clay : clay content     (%)
    BD   : bulk density     (g cm⁻³)
    θs   : saturated VWC    (cm³ cm⁻³)  mean of 1000 bootstrap NN members
    θr   : residual VWC     (cm³ cm⁻³)  mean of 1000 bootstrap NN members
    α    : van Genuchten α  (cm⁻¹)      natural units (NN outputs log10 α)
    n    : van Genuchten n  (-)         natural units (NN outputs log10 n)

Notes
-----
Rosetta soildata column order (official):
    [sand%, silt%, clay%, BD (g/cm3), th33, th1500]
    sand/silt/clay required; BD and water contents optional.
    The "best-available" model is selected automatically per pixel.

"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xarray as xr
import threadpoolctl
from rosetta import Rosetta

_DEFAULT_BATCH_SIZE = 10_000


def _make_rosetta(rosetta_version: int) -> Rosetta:
    """Instantiate Rosetta once — loads NN weights from disk a single time."""
    return Rosetta(rosetta_version=rosetta_version, model_code=3)


# ---------------------------------------------------------------------------
# Rosetta wrapper — one depth slice at a time
# ---------------------------------------------------------------------------

def run_rosetta_layer(sand_2d, silt_2d, clay_2d, bd_2d,
                      rose: Rosetta,
                      batch_size: int = _DEFAULT_BATCH_SIZE):
    """
    Call Rosetta for one depth layer in memory- and compute-efficient batches.

    Safe to call from multiple threads simultaneously — rose.predict() is
    stateless. Each call limits OpenBLAS to 1 thread so parallel workers
    do not contend for CPU cores.

    Parameters
    ----------
    sand_2d, silt_2d, clay_2d : 2-D numpy arrays (%) — NOT dask arrays.
        Call .to_numpy() before passing.
    bd_2d      : 2-D numpy array (g/cm3)
    rose       : Rosetta instance (shared, read-only)
    batch_size : rows per predict() call. Keep at ~10,000 for best
                 numpy/BLAS performance. Larger values saturate memory
                 bandwidth and slow down dramatically.

    Returns
    -------
    params : dict with 2-D float32 arrays:
        'thr'   residual VWC        (m3/m3)
        'ths'   saturated VWC       (m3/m3)
        'alpha' van Genuchten alpha  (1/cm)  — natural units, not log10
        'n'     van Genuchten n      (-)     — natural units, not log10
    valid  : 2-D bool — True where Rosetta returned a result
    """
    shape = sand_2d.shape

    # ------------------------------------------------------------------
    # 1. Identify valid pixels (skip NaN entirely)
    # ------------------------------------------------------------------
    nan_mask = (
            np.isnan(sand_2d) | np.isnan(silt_2d) |
            np.isnan(clay_2d) | np.isnan(bd_2d)
    )
    valid_idx = np.flatnonzero(~nan_mask)
    n_valid = valid_idx.size

    n_batches = int(np.ceil(n_valid / batch_size))
    logging.info(
        f"  Rosetta model_code=3: {n_valid:,} valid pixels, "
        f"{n_batches} batches of {batch_size:,}"
    )

    if n_valid == 0:
        out = {k: np.full(shape, np.nan, dtype=np.float32)
               for k in ("thr", "ths", "alpha", "n")}
        return out, np.zeros(shape, dtype=bool)

    # ------------------------------------------------------------------
    # 2. Extract valid pixels, normalise SSC, pre-stack into (N, 4)
    # ------------------------------------------------------------------
    ssc = np.stack([
        sand_2d.ravel()[valid_idx],
        silt_2d.ravel()[valid_idx],
        clay_2d.ravel()[valid_idx],
    ], axis=1).astype(np.float64)  # (N, 3)

    ssc_sum = ssc.sum(axis=1, keepdims=True).clip(min=1.0)
    ssc /= ssc_sum
    ssc *= 100.0

    soildata = np.empty((n_valid, 4), dtype=np.float64)
    soildata[:, :3] = ssc
    soildata[:, 3] = bd_2d.ravel()[valid_idx].astype(np.float64)
    del ssc

    # ------------------------------------------------------------------
    # 3. Batched predict() calls.
    #    BLAS is limited to 1 thread per worker so that parallel depth
    #    workers share CPU cores rather than contending for them.
    # ------------------------------------------------------------------
    thr_c = np.empty(n_valid, dtype=np.float32)
    ths_c = np.empty(n_valid, dtype=np.float32)
    alpha_c = np.empty(n_valid, dtype=np.float32)
    n_c = np.empty(n_valid, dtype=np.float32)

    for b in range(n_batches):
        s = b * batch_size
        e = min(s + batch_size, n_valid)

        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            retc_boot, _ = rose.predict(soildata[s:e])  # (1000, batch, 4)

        mean_boot = np.mean(retc_boot, axis=0)  # (batch, 4)
        del retc_boot

        thr_c[s:e] = mean_boot[:, 0]
        ths_c[s:e] = mean_boot[:, 1]
        alpha_c[s:e] = np.power(10.0, mean_boot[:, 2])  # log10 -> natural
        n_c[s:e] = np.power(10.0, mean_boot[:, 3])  # log10 -> natural
        del mean_boot

        if n_batches > 1:
            logging.info(f"    batch {b + 1}/{n_batches} done")

    del soildata

    # ------------------------------------------------------------------
    # 4. Scatter results back to 2-D output arrays
    # ------------------------------------------------------------------
    out = {k: np.full(shape, np.nan, dtype=np.float32)
           for k in ("thr", "ths", "alpha", "n")}
    rows, cols = np.unravel_index(valid_idx, shape)
    out["thr"][rows, cols] = thr_c
    out["ths"][rows, cols] = ths_c
    out["alpha"][rows, cols] = alpha_c
    out["n"][rows, cols] = n_c

    return out, ~np.isnan(out["thr"])


# ---------------------------------------------------------------------------
# Per-depth worker
# ---------------------------------------------------------------------------

def _process_depth(d, ds, rose, batch_size):
    """Extract arrays for one depth, run Rosetta, return labelled DataArrays."""
    logging.info(f"Processing depth {d} cm ...")
    sl = ds.sel(depth=d)

    params, _valid = run_rosetta_layer(
        sand_2d=sl["sand"].to_numpy().astype(np.float64),
        silt_2d=sl["silt"].to_numpy().astype(np.float64),
        clay_2d=sl["clay"].to_numpy().astype(np.float64),
        bd_2d=sl["trd"].to_numpy().astype(np.float64),
        rose=rose,
        batch_size=batch_size,
    )

    das = {
        k: (
            xr.DataArray(arr, coords={"y": sl.y, "x": sl.x}, dims=["y", "x"])
            .astype(np.float32)
            .assign_coords(depth=int(d))
        )
        for k, arr in params.items()
    }
    return int(d), das


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_zhang_schaap(ds, rosetta_version=3,
                       batch_size=_DEFAULT_BATCH_SIZE,
                       max_workers=None):
    """
    Zhang & Schaap (2017) Rosetta3 PTF — process all depth layers in
    parallel and run the Rosetta neural network ensemble.

    Expected variables in ds:
        sand  (%)      : sand content
        clay  (%)      : clay content
        trd   (g cm⁻³) : bulk density

    New variables added to ds:
        thr, ths, alpha, n   — MvG parameters   (depth, y, x)

    ds.load() must be called before this function.

    Parameters:
        ds              : xarray.Dataset — already loaded into memory
        rosetta_version : 1, 2, or 3
        batch_size      : pixels per Rosetta.predict() call.
                          Default 10,000 is the empirical BLAS sweet spot.
                          Do not raise above ~20,000.
        max_workers     : parallel depth workers.
                          None → one worker per depth layer.
                          Peak RAM ≈ max_workers × batch_size × 400 MB / 10,000.
                          At 6 workers × batch_size=10,000: ~2.4 GB.
    """
    ds["silt"] = (100.0 - ds["sand"] - ds["clay"]).clip(0, 100).astype(np.float32)

    # Single Rosetta instance shared across all workers — predict() is stateless
    rose = _make_rosetta(rosetta_version)
    logging.info(f"Rosetta v{rosetta_version} model loaded (model_code=3).")

    depths = list(ds.depth.values)
    if max_workers is None:
        max_workers = len(depths)

    param_layers: dict[str, list] = {k: [] for k in ("thr", "ths", "alpha", "n")}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_depth, d, ds, rose, batch_size): d
            for d in depths
        }
        results = {}
        for future in as_completed(futures):
            depth_val, das = future.result()
            results[depth_val] = das

    # Reassemble in original depth order
    for d in depths:
        for k in param_layers:
            param_layers[k].append(results[int(d)][k])

    for k, layers in param_layers.items():
        ds[k] = xr.concat(layers, dim="depth")

    return ds


if __name__ == "__main__":
    pass
