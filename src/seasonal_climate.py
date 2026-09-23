"""
Seasonal climate metrics from Meteotest daily NetCDF data.

Produces, per target year:
  - pr_{year}_DJFMAM.tif     precipitation total, Dec(year-1) + Jan-May(year)
  - pr_{year}_JJA.tif        precipitation total, Jun-Aug(year)
  - tas_{year}_JJA.tif       mean temperature, Jun-Aug(year)
  - tasmax_{year}_JJA.tif    max temperature, Jun-Aug(year)
  - vpd_{year}_JJA.tif       mean VPD, Jun-Aug(year)
  - vpdmax_{year}_JJA.tif    max VPD, Jun-Aug(year)

VPD (Vapor Pressure Deficit, kPa) is computed at native daily resolution
from tas + td (Tetens formula), then averaged over JJA.

calculate_baseline_climatology() additionally produces per-pixel seasonal
baselines (mean + sample std across years) over a reference period, e.g.
  - pr_baseline_mean_DJFMAM_1991-2020.tif / pr_baseline_std_DJFMAM_1991-2020.tif
  - pr_baseline_mean_JJA_1991-2020.tif / pr_baseline_std_JJA_1991-2020.tif
  - tas_baseline_mean_JJA_1991-2020.tif / tas_baseline_std_JJA_1991-2020.tif

All outputs are saved as Cloud-Optimized GeoTIFFs (COG).
"""
import logging
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS as PyProjCRS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

# Season definitions: which calendar months, and whether December belongs
# to the PREVIOUS calendar year.
SEASONS = {
    "DJFMAM": {"months": [12, 1, 2, 3, 4, 5], "dec_in_prev_year": True},
    "JJA": {"months": [6, 7, 8], "dec_in_prev_year": False},
}

# Which seasons to baseline per variable, and how to reduce each season
# (must match the `how` used in calculate_seasonal_meteo for that variable).
BASELINE_SEASONS = {
    "pr": {"DJFMAM": "sum", "JJA": "sum"},
    "tas": {"JJA": "mean"},
}

# Below this many valid years, still compute the baseline but flag it as
# unreliable rather than silently returning it.
MIN_BASELINE_YEARS = 10

DIM_RENAME = {
    "chx": "x", "lon": "x", "longitude": "x", "e": "x", "east": "x",
    "chy": "y", "lat": "y", "latitude": "y", "n": "y", "north": "y",
}

TARGET_CRS = "EPSG:3035"

# =============================================================================
# SPATIAL HELPERS
# =============================================================================
def _parse_crs(da, ds, path):
    """Extract CRS from common NetCDF metadata patterns, falling back to CH1903+."""
    if "esri_pe_string" in da.attrs:
        try:
            return da.rio.write_crs(PyProjCRS.from_wkt(da.attrs["esri_pe_string"]))
        except Exception as e:
            log.warning(f"Failed to parse esri_pe_string in {path.name}: {e}")

    if "grid_mapping" in da.attrs and da.attrs["grid_mapping"] in ds:
        grid = ds[da.attrs["grid_mapping"]]
        for attr in ("crs_wkt", "spatial_ref", "esri_pe_string"):
            if attr in grid.attrs:
                try:
                    return da.rio.write_crs(PyProjCRS.from_wkt(grid.attrs[attr]))
                except Exception as e:
                    log.warning(f"Failed to parse grid_mapping.{attr} in {path.name}: {e}")

    log.warning(f"Falling back to EPSG:2056 (CH1903+/LV95) for {path.name}")
    return da.rio.write_crs("EPSG:2056")


def _apply_spatial_fixes(da, ds, path):
    """Rename spatial dims to x/y, add band dim, attach CRS."""
    dim_rename = {d: DIM_RENAME[d.lower()] for d in da.dims if d.lower() in DIM_RENAME}
    if dim_rename:
        da = da.rename(dim_rename)
    da = _parse_crs(da, ds, path)
    return da.rio.set_spatial_dims(x_dim="x", y_dim="y")


# =============================================================================
# LOADING
# =============================================================================
def load_variable(input_dir, var_name, years, chunks="auto"):
    """
    Lazily open and concatenate every NetCDF file in `input_dir` that
    contains data for any of `years`, at native (daily) resolution.

    Returns a dask-backed DataArray: no data is actually read from disk
    here. Values are only realized downstream when a small seasonal
    reduction (sum/mean/max) is computed, so the full multi-year daily
    stack never has to fit in memory at once.
    """
    input_dir = Path(input_dir)
    nc_files = sorted(input_dir.rglob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {input_dir}")

    arrays = []
    crs = None
    for path in nc_files:
        if not any(str(y) in path.name for y in years):
            continue

        ds = xr.open_dataset(path, chunks=chunks)
        var = var_name if var_name in ds else list(ds.data_vars)[0]
        da = _apply_spatial_fixes(ds[var], ds, path)
        da = da.sel(time=da.time.dt.year.isin(years))

        if len(da.time) == 0:
            ds.close()
            continue

        if crs is None:
            crs = da.rio.crs
        arrays.append(da)
        ds.close()

    if not arrays:
        raise ValueError(f"No data found for years {years} in {input_dir}")

    combined = xr.concat(arrays, dim="time", coords="minimal", compat="override")

    # Avoid combined.sortby("time"): xarray's sortby aligns via a deep-copying
    # align() call internally, which transiently doubles memory for large
    # arrays (OOM risk). Reorder in place with isel instead, and only if the
    # time axis isn't already sorted.
    time_vals = combined["time"].values
    if not np.all(time_vals[:-1] <= time_vals[1:]):
        order = np.argsort(time_vals, kind="stable")
        combined = combined.isel(time=order)

    log.info(f"Opened '{var_name}' lazily: {len(combined.time)} daily steps, years {years}")
    return combined.rio.write_crs(crs)


# =============================================================================
# SEASONAL SELECTION & AGGREGATION
# =============================================================================
def select_season(da, year, season_cfg):
    """
    Select the timesteps belonging to `year`'s season, handling the
    Dec(year-1) -> following-year convention where applicable.
    """
    months = season_cfg["months"]
    if season_cfg["dec_in_prev_year"] and 12 in months:
        dec_mask = (da.time.dt.year == year - 1) & (da.time.dt.month == 12)
        other_months = [m for m in months if m != 12]
        rest_mask = (da.time.dt.year == year) & (da.time.dt.month.isin(other_months))
        mask = dec_mask | rest_mask
    else:
        mask = (da.time.dt.year == year) & (da.time.dt.month.isin(months))
    return da.sel(time=mask)


def seasonal_aggregate(da, year, season_cfg, how):
    """
    how: 'sum', 'mean', or 'max'.

    Computes (materializes) the reduction here rather than leaving it lazy:
    `sel` may span several years' worth of dask-backed daily data, but the
    reduced result is a single small 2D field, so this is the point where
    it's cheapest to pull data off disk and into memory.
    """
    sel = select_season(da, year, season_cfg)
    if len(sel.time) == 0:
        log.warning(f"No timesteps for {year} ({season_cfg['months']}), skipping")
        return None
    return getattr(sel, how)(dim="time", skipna=False).compute()


# =============================================================================
# VPD (Vapor Pressure Deficit)
# =============================================================================
def saturation_vapour_pressure(temp_c):
    """Tetens formula: es(T) = 0.6108 * exp(17.27*T / (T+237.3))  [kPa]"""
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def compute_vpd_daily(tas, td):
    """VPD = es(Tair) - es(Tdew), clamped to >= 0. Must be computed BEFORE
    any temporal averaging (i.e. at native daily resolution)."""
    vpd = saturation_vapour_pressure(tas) - saturation_vapour_pressure(td)
    return vpd.clip(min=0.0)


# =============================================================================
# SAVING
# =============================================================================
def save_geotiff(da, out_path, source_crs, target_crs=TARGET_CRS):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    da = da.rio.write_crs(source_crs)
    if "band" not in da.dims:
        da = da.expand_dims("band")
    if str(da.rio.crs) != target_crs:
        da = da.rio.reproject(target_crs)

    da = da.astype("float32").rio.write_crs(target_crs)
    da = da.rio.write_nodata(np.nan)
    da.rio.to_raster(str(out_path), driver="COG", compress="deflate", OVERVIEWS="AUTO")
    log.info(f"Saved {out_path}")


def _maybe_process(da, year, season_cfg, how, out_path, source_crs):
    """Skip the (expensive, disk-reading) aggregation entirely if the output
    already exists, instead of computing it and only then discovering it's
    not needed."""
    if out_path.exists():
        log.info(f"Skipping {out_path.name}, already exists.")
        return
    result = seasonal_aggregate(da, year, season_cfg, how)
    if result is not None:
        save_geotiff(result, out_path, source_crs)


def calculate_seasonal_meteo(input_dirs, output_dir, target_years):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load one extra year up front so DJFMAM's December is always available.
    load_years = sorted(set(target_years) | {y - 1 for y in target_years})

    # These are dask-backed (lazy): opening them doesn't read any data.
    pr = load_variable(input_dirs["pr"], "pr", load_years)
    source_crs = pr.rio.crs
    tas = load_variable(input_dirs["tas"], "tas", load_years)
    tasmax = load_variable(input_dirs["tasmax"], "tasmax", load_years)
    td = load_variable(input_dirs["td"], "td", load_years)

    log.info("Building lazy daily VPD graph...")
    vpd_daily = compute_vpd_daily(tas, td).rio.write_crs(source_crs)

    for year in target_years:
        log.info(f"--- Year {year} ---")

        _maybe_process(pr, year, SEASONS["DJFMAM"], "sum",
                        output_dir / f"pr_{year}_DJFMAM.tif", source_crs)
        _maybe_process(pr, year, SEASONS["JJA"], "sum",
                        output_dir / f"pr_{year}_JJA.tif", source_crs)
        _maybe_process(tas, year, SEASONS["JJA"], "mean",
                        output_dir / f"tas_{year}_JJA.tif", source_crs)
        _maybe_process(tasmax, year, SEASONS["JJA"], "max",
                        output_dir / f"tasmax_{year}_JJA.tif", source_crs)
        _maybe_process(vpd_daily, year, SEASONS["JJA"], "mean",
                        output_dir / f"vpd_{year}_JJA.tif", source_crs)
        _maybe_process(vpd_daily, year, SEASONS["JJA"], "max",
                        output_dir / f"vpdmax_{year}_JJA.tif", source_crs)

    log.info("Done.")


# =============================================================================
# BASELINE CLIMATOLOGY
# =============================================================================
def _maybe_process_baseline(da, years, season_cfg, how, out_mean_path, out_std_path, source_crs):
    """Skip entirely if both outputs already exist; otherwise compute the
    per-year seasonal aggregate for every baseline year, then reduce across
    years to a mean and a (sample, ddof=1) std."""
    if out_mean_path.exists() and out_std_path.exists():
        log.info(f"Skipping {out_mean_path.name} / {out_std_path.name}, already exist.")
        return

    yearly, years_used = [], []
    for year in years:
        agg = seasonal_aggregate(da, year, season_cfg, how)
        if agg is None:
            continue
        yearly.append(agg)
        years_used.append(year)

    if not yearly:
        log.warning(f"No years available for {out_mean_path.name}, skipping.")
        return
    if len(years_used) < MIN_BASELINE_YEARS:
        log.warning(
            f"{out_mean_path.name}: only {len(years_used)} baseline years available "
            f"(minimum recommended: {MIN_BASELINE_YEARS}). Years used: {years_used}"
        )

    stacked = xr.concat(yearly, dim="year").assign_coords(year=("year", years_used))

    baseline_mean = stacked.mean(dim="year", skipna=False)
    save_geotiff(baseline_mean, out_mean_path, source_crs)

    baseline_std = stacked.std(dim="year", ddof=1, skipna=False)
    save_geotiff(baseline_std, out_std_path, source_crs)


def calculate_baseline_climatology(input_dirs, output_dir, baseline_start=1991, baseline_end=2020):
    """
    Per-pixel seasonal climatology (mean + sample std across years) over
    [baseline_start, baseline_end], for every season in BASELINE_SEASONS.

    Defaults to the WMO 1991-2020 normal period. DJFMAM's December belongs to
    the previous calendar year, so December of (baseline_start - 1) is loaded
    and included automatically.

    Outputs, per variable/season:
        {var}_baseline_mean_{season}_{baseline_start}-{baseline_end}.tif
        {var}_baseline_std_{season}_{baseline_start}-{baseline_end}.tif
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    years = list(range(baseline_start, baseline_end + 1))
    load_years = list(range(baseline_start - 1, baseline_end + 1))

    for var, seasons in BASELINE_SEASONS.items():
        log.info(f"--- Baseline: {var} ---")
        da = load_variable(input_dirs[var], var, load_years)
        source_crs = da.rio.crs

        for season, how in seasons.items():
            suffix = f"{season}_{baseline_start}-{baseline_end}"
            _maybe_process_baseline(
                da, years, SEASONS[season], how,
                output_dir / f"{var}_baseline_mean_{suffix}.tif",
                output_dir / f"{var}_baseline_std_{suffix}.tif",
                source_crs,
            )

    log.info("Baseline climatology done.")


if __name__ == "__main__":
    pass