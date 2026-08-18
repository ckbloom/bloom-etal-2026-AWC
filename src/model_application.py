"""
apply_dsi_multiyear.py
======================
Apply a trained drought-susceptibility logistic regression model (saved as
dsi_result.pkl by drought_susceptibility_index.run_dsi_pipeline) to multiple
years of input rasters, then produce:

  1. Per-year discoloration probability rasters (.tif).
  2. Comparative box-and-whisker plots of predicted-probability distributions,
     split by discolored / not-discolored pixels (separability analysis).
  3. A violin + strip overlay variant for richer distributional detail.
  4. An ROC / AUC curve per year to quantify separability numerically.
  5. A summary CSV with per-year separability statistics.
"""

from __future__ import annotations

import csv
import logging
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression

# Import the feature-scaling helper from the pipeline script so test/new
# rasters are transformed identically to how the training data was scaled.
from src.dsi import build_feature_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*angle from rectified to skew grid parameter lost.*",
    module="pyproj",
)


@dataclass
class YearResult:
    """Predicted-probability arrays and separability metrics for one year."""
    label: str                          # e.g. "2018"
    prob: np.ndarray                    # flat array of predicted probabilities (valid pixels)
    disco_mask: np.ndarray              # bool, True = discolored pixel
    # Separability metrics
    auc: float = 0.0
    mw_u: float = 0.0
    mw_p: float = 1.0
    cohens_d: float = 0.0
    mean_disco: float = 0.0
    mean_nondisco: float = 0.0
    median_disco: float = 0.0
    median_nondisco: float = 0.0
    n_disco: int = 0
    n_nondisco: int = 0


def _load_and_apply(
        awc_path: str,
        disco_path: str,
        norm_params: dict,
        model: LogisticRegression,
        extra_paths: list[str] | None = None,
        disco_threshold: float = 4000.0,
        output_tif: str | None = None,
) -> YearResult | None:
    """
    Load rasters for one year, apply the pre-fitted logistic regression model,
    and optionally export a discoloration-probability GeoTIFF.

    Returns None if loading fails (so a missing year is skipped gracefully).
    """
    extra_paths = extra_paths or []

    try:
        awc = rioxarray.open_rasterio(awc_path, masked=True)
    except Exception as exc:
        logging.error(f"  Cannot open AWC raster '{awc_path}': {exc}")
        return None

    try:
        disco_da = rioxarray.open_rasterio(disco_path, masked=True)
        disco_r = disco_da.rio.reproject_match(awc, resampling=Resampling.average)
    except Exception as exc:
        logging.error(f"  Cannot open/reproject disco raster '{disco_path}': {exc}")
        return None

    awc_arr = awc.values.astype(np.float32).squeeze()
    disco_arr = disco_r.values.astype(np.float32).squeeze()

    extra_arrs: list[np.ndarray] = []
    for path in extra_paths:
        try:
            da = rioxarray.open_rasterio(path, masked=True)
            da_r = da.rio.reproject_match(awc, resampling=Resampling.bilinear)
            extra_arrs.append(da_r.values.astype(np.float32).squeeze())
        except Exception as exc:
            logging.error(f"  Cannot open/reproject extra raster '{path}': {exc}")
            return None

    # Valid-pixel mask (no NaNs in any layer)
    valid = ~np.isnan(awc_arr) & ~np.isnan(disco_arr)
    for arr in extra_arrs:
        valid &= ~np.isnan(arr)

    if valid.sum() == 0:
        logging.warning("  No valid pixels — skipping year.")
        return None

    # Scale features using the TRAINING norm_params (no re-fitting), then get
    # probabilities directly from the fitted logistic regression model.
    X_valid, _ = build_feature_matrix(
        awc_arr[valid],
        [arr[valid] for arr in extra_arrs],
        norm_params=norm_params,
    )
    prob_flat = model.predict_proba(X_valid)[:, 1].astype(np.float32)
    disco_bin = (disco_arr[valid] >= disco_threshold).astype(bool)

    logging.info(
        f"  Valid pixels: {valid.sum():,}  |  "
        f"Discolored: {disco_bin.sum():,} ({100 * disco_bin.mean():.1f} %)"
    )

    # Optionally export probability raster
    if output_tif:
        prob_full = np.full(awc_arr.shape, np.nan, dtype=np.float32)
        prob_full[valid] = prob_flat

        prob_da = xr.DataArray(
            prob_full[np.newaxis, :, :], dims=awc.dims, coords=awc.coords
        )
        prob_da.rio.write_crs(awc.rio.crs, inplace=True)
        prob_da = prob_da.fillna(-9999.0)
        prob_da.rio.write_nodata(-9999.0, inplace=True)
        prob_da.name = "discoloration_probability"
        prob_da.rio.to_raster(output_tif)
        logging.info(f"  Probability raster → {output_tif}")

    return YearResult(
        label=str(Path(disco_path).stem),   # overwritten by caller
        prob=prob_flat,
        disco_mask=disco_bin,
    )


def _compute_separability(yr: YearResult) -> YearResult:
    """Fill in separability statistics on a YearResult in-place."""
    d = yr.prob[yr.disco_mask]
    nd = yr.prob[~yr.disco_mask]

    yr.n_disco = int(d.size)
    yr.n_nondisco = int(nd.size)
    yr.mean_disco = float(np.mean(d)) if d.size else float("nan")
    yr.mean_nondisco = float(np.mean(nd)) if nd.size else float("nan")
    yr.median_disco = float(np.median(d)) if d.size else float("nan")
    yr.median_nondisco = float(np.median(nd)) if nd.size else float("nan")

    if d.size > 1 and nd.size > 1:
        u_stat, p_val = mannwhitneyu(d, nd, alternative="greater")
        yr.mw_u = float(u_stat)
        yr.mw_p = float(p_val)

        # AUC from Mann-Whitney U
        yr.auc = float(u_stat / (d.size * nd.size))

        # Cohen's d
        pooled_std = np.sqrt(
            (np.std(d, ddof=1) ** 2 + np.std(nd, ddof=1) ** 2) / 2.0
        )
        yr.cohens_d = float((np.mean(d) - np.mean(nd)) / pooled_std) if pooled_std > 0 else float("nan")

    logging.info(
        f"  [{yr.label}] AUC={yr.auc:.3f}  Cohen's d={yr.cohens_d:.3f}  "
        f"p(MW)={yr.mw_p:.2e}"
    )
    return yr


_COLOR_DISCO = "#ca6702"
_COLOR_NONDISCO = "#005f73"
_ALPHA_FILL = 0.85

def _subsample(arr: np.ndarray, max_pts: int = 5_000, seed: int = 42) -> np.ndarray:
    """Randomly subsample an array for strip/jitter plots."""
    if arr.size <= max_pts:
        return arr
    rng = np.random.default_rng(seed)
    return rng.choice(arr, size=max_pts, replace=False)


def plot_boxwhisker_comparison(
        year_results: list[YearResult],
        x_label: str = "Predicted Probability",
        title: str | None = None,
        output_path: str | None = None,
        max_strip_pts: int = 3_000,
) -> None:
    """
    Grouped box-and-whisker plot: for each year, side-by-side boxes for
    discolored (orange) and not-discolored (teal) pixels.
    """
    n_years = len(year_results)
    fig, ax = plt.subplots(figsize=(max(8, 2.5 * n_years), 6))

    group_gap = 1.0          # distance between year groups
    box_width = 0.30
    offset = 0.20             # half-distance between the two boxes in a group

    positions_disco, positions_nondisco = [], []
    data_disco, data_nondisco = [], []
    xtick_positions, xtick_labels = [], []

    for i, yr in enumerate(year_results):
        x_center = i * group_gap
        pos_d = x_center - offset
        pos_nd = x_center + offset

        positions_disco.append(pos_d)
        positions_nondisco.append(pos_nd)
        data_disco.append(yr.prob[yr.disco_mask])
        data_nondisco.append(yr.prob[~yr.disco_mask])
        xtick_positions.append(x_center)
        xtick_labels.append(yr.label)

    bp_d = ax.boxplot(
        data_disco, positions=positions_disco, widths=box_width,
        patch_artist=True, notch=False,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker=".", markersize=2, alpha=0.3, markerfacecolor=_COLOR_DISCO),
        boxprops=dict(facecolor=_COLOR_DISCO, alpha=_ALPHA_FILL),
    )
    bp_nd = ax.boxplot(
        data_nondisco, positions=positions_nondisco, widths=box_width,
        patch_artist=True, notch=False,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker=".", markersize=2, alpha=0.3, markerfacecolor=_COLOR_NONDISCO),
        boxprops=dict(facecolor=_COLOR_NONDISCO, alpha=_ALPHA_FILL),
    )

    # Annotate AUC above each group
    for i, yr in enumerate(year_results):
        x_center = i * group_gap
        ax.text(
            x_center, 1.01,
            f"AUC={yr.auc:.2f}\nd={yr.cohens_d:.2f}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8.5,
            color="#333333",
        )

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=11)
    ax.set_ylabel(x_label, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_xlim(-0.6, (n_years - 1) * group_gap + 0.6)

    patch_d = mpatches.Patch(color=_COLOR_DISCO, alpha=_ALPHA_FILL, label="Discolored")
    patch_nd = mpatches.Patch(color=_COLOR_NONDISCO, alpha=_ALPHA_FILL, label="Not discolored")
    ax.legend(handles=[patch_d, patch_nd], fontsize=11, loc="upper left")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
        logging.info(f"Box-whisker plot saved → {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_violin_comparison(
        year_results: list[YearResult],
        x_label: str = "Predicted Probability",
        title: str | None = None,
        output_path: str | None = None,
        max_strip_pts: int = 2_000,
) -> None:
    """
    Violin plot variant with a jitter strip overlay for raw data density.
    """
    n_years = len(year_results)
    fig, ax = plt.subplots(figsize=(max(8, 2.8 * n_years), 6))

    group_gap = 1.0
    offset = 0.22
    violin_width = 0.36
    rng = np.random.default_rng(0)

    for i, yr in enumerate(year_results):
        x_center = i * group_gap
        for sign, arr, color, label in [
            (-1, yr.prob[yr.disco_mask],  _COLOR_DISCO,   "Discolored"),
            (+1, yr.prob[~yr.disco_mask], _COLOR_NONDISCO, "Not discolored"),
        ]:
            pos = x_center + sign * offset
            if arr.size < 4:
                continue
            vp = ax.violinplot(
                arr, positions=[pos], widths=violin_width,
                showmedians=True, showextrema=False,
            )
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.55)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(1.8)

            # Jitter strip
            sub = _subsample(arr, max_pts=max_strip_pts)
            jitter = rng.uniform(-0.06, 0.06, size=sub.size)
            ax.scatter(
                pos + jitter, sub,
                s=1.5, color=color, alpha=0.18, linewidths=0,
            )

        # AUC annotation
        ax.text(
            x_center, 1.01,
            f"AUC={yr.auc:.2f}\nd={yr.cohens_d:.2f}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8.5,
            color="#333333",
        )

    ax.set_xticks([i * group_gap for i in range(n_years)])
    ax.set_xticklabels([yr.label for yr in year_results], fontsize=11)
    ax.set_ylabel(x_label, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_xlim(-0.6, (n_years - 1) * group_gap + 0.6)

    patch_d = mpatches.Patch(color=_COLOR_DISCO, alpha=0.7, label="Discolored")
    patch_nd = mpatches.Patch(color=_COLOR_NONDISCO, alpha=0.7, label="Not discolored")
    ax.legend(handles=[patch_d, patch_nd], fontsize=11, loc="upper left")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
        logging.info(f"Violin plot saved → {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_roc_curves(
        year_results: list[YearResult],
        title: str | None = None,
        output_path: str | None = None,
) -> None:
    """
    Per-year ROC curves derived from the predicted probability as the ranking
    score. The area under each curve equals the Mann-Whitney AUC stored in
    YearResult.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.get_cmap("tab10", len(year_results))

    for i, yr in enumerate(year_results):
        if yr.n_disco == 0 or yr.n_nondisco == 0:
            continue

        # Build ROC by sorting on predicted probability
        scores = yr.prob
        labels = yr.disco_mask.astype(int)
        order = np.argsort(scores)[::-1]   # descending
        sorted_labels = labels[order]

        tp = np.cumsum(sorted_labels)
        fp = np.cumsum(1 - sorted_labels)
        tpr = tp / tp[-1]
        fpr = fp / fp[-1]

        ax.plot(
            fpr, tpr,
            color=cmap(i), linewidth=1.8,
            label=f"{yr.label}  (AUC={yr.auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
        logging.info(f"ROC plot saved → {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_dsi_histograms(
        year_results: list[YearResult],
        x_label: str = "Predicted Probability",
        title: str | None = None,
        output_path: str | None = None,
        n_bins: int = 50,
) -> None:
    """
    Small-multiples histogram grid: one panel per year, overlaid disco / non-disco.
    Useful for inspecting distribution shape and overlap.
    """
    n = len(year_results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, yr in enumerate(year_results):
        ax = axes[idx // ncols][idx % ncols]
        bins = np.linspace(0, 1, n_bins + 1)

        ax.hist(yr.prob[~yr.disco_mask], bins=bins, color=_COLOR_NONDISCO,
                alpha=0.55, density=True, label="Not discolored")
        ax.hist(yr.prob[yr.disco_mask], bins=bins, color=_COLOR_DISCO,
                alpha=0.65, density=True, label="Discolored")

        ax.set_title(
            f"{yr.label}  (AUC={yr.auc:.2f}, d={yr.cohens_d:.2f})",
            fontsize=10,
        )
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logging.info(f"Histogram grid saved → {output_path}")
    else:
        plt.show()
    plt.close(fig)


def export_summary_csv(year_results: list[YearResult], output_path: str) -> None:
    """Write per-year separability statistics to a CSV."""
    fieldnames = [
        "year", "n_disco", "n_nondisco",
        "mean_prob_disco", "mean_prob_nondisco",
        "median_prob_disco", "median_prob_nondisco",
        "auc", "cohens_d", "mw_u", "mw_p",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for yr in year_results:
            writer.writerow({
                "year": yr.label,
                "n_disco": yr.n_disco,
                "n_nondisco": yr.n_nondisco,
                "mean_prob_disco": f"{yr.mean_disco:.6f}",
                "mean_prob_nondisco": f"{yr.mean_nondisco:.6f}",
                "median_prob_disco": f"{yr.median_disco:.6f}",
                "median_prob_nondisco": f"{yr.median_nondisco:.6f}",
                "auc": f"{yr.auc:.6f}",
                "cohens_d": f"{yr.cohens_d:.6f}",
                "mw_u": f"{yr.mw_u:.2f}",
                "mw_p": f"{yr.mw_p:.4e}",
            })
    logging.info(f"Summary CSV → {output_path}")


def _dump_pkl_fields(pkl_path: str) -> None:
    """
    Diagnostic: print the top-level keys stored in a dsi_result.pkl, plus the
    keys nested under norm_params. Call this manually if you get a KeyError
    after loading a pkl produced by an unfamiliar pipeline version.

    Current (spatial-CV) pipeline: top-level keys include "norm_params",
    "final_model", "feature_names", "fold_rows", "cv_summary",
    "pooled_calibration_r2", "oof_p", "y", "multicollinearity", "run_config".
    Older (checkerboard train/test) pipeline versions instead had a nested
    "fit_result" dict with a ["model"] key -- this script no longer supports
    those pkls directly (see the "final_model" fix in _load_and_apply).
    """
    with open(pkl_path, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict):
        logging.info(f"--- dsi_result.pkl top-level keys: {list(obj.keys())}")
        norm_params = obj.get("norm_params")
        if isinstance(norm_params, dict):
            logging.info(f"--- norm_params keys: {list(norm_params.keys())}")
        if "fit_result" in obj:
            logging.warning(
                "  This pkl has a legacy 'fit_result' key (old checkerboard-split "
                "pipeline) -- update it with run_dsi_pipeline() from the current "
                "drought_susceptibility_index.py before using this script."
            )
    else:
        logging.info(f"--- dsi_result.pkl is a {type(obj)}: {vars(obj).keys()}")


def _build_x_label(feature_names: list[str]) -> str:
    """Build a readable axis label listing the model's predictive features."""
    return "Predicted Probability of Discoloration (" + ", ".join(feature_names) + ")"


@dataclass
class YearSpec:
    """Specification for one year's dataset."""
    label: str                          # e.g. "2018"
    awc_path: str                       # AWC raster (can be the same for all years)
    disco_path: str                     # Discoloration raster for this year
    extra_paths: list[str] = field(default_factory=list)  # Additional rasters (e.g. precip)


def apply_dsi_to_years(
        pkl_path: str,
        year_specs: list[YearSpec],
        output_dir: str,
        disco_threshold: float = 4000.0,
        x_label: str | None = None,
        export_rasters: bool = True,
        max_strip_pts: int = 3_000,
) -> list[YearResult]:
    """
    Load a trained model result from pkl_path, apply it to each YearSpec, and
    produce all comparative plots + a summary CSV.

    Parameters
    ----------
    pkl_path        : Path to dsi_result.pkl produced by run_dsi_pipeline().
    year_specs      : List of YearSpec objects, one per year / scenario.
    output_dir      : Root directory for all outputs.
    disco_threshold : Raw discoloration value above which a pixel is "discolored".
    x_label         : X-axis label for probability plots.
    export_rasters  : If True, write a probability .tif for every year.
    max_strip_pts   : Max points in violin jitter strips (for speed).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load trained model result (plain dict, as saved by run_dsi_pipeline)
    logging.info(f"Loading trained DSI model from '{pkl_path}' …")
    with open(pkl_path, "rb") as fh:
        dsi_result: dict = pickle.load(fh)

    norm_params: dict = dsi_result["norm_params"]
    model: LogisticRegression = dsi_result["final_model"]
    feature_names: list[str] = dsi_result.get("feature_names", norm_params["feature_names"])

    if x_label is None:
        x_label = _build_x_label(feature_names)

    # Process each year
    year_results: list[YearResult] = []
    for spec in year_specs:
        logging.info(f"\nProcessing year/scenario: {spec.label} …")
        tif_path = str(out / f"prob_{spec.label}.tif") if export_rasters else None

        yr = _load_and_apply(
            awc_path=spec.awc_path,
            disco_path=spec.disco_path,
            norm_params=norm_params,
            model=model,
            extra_paths=spec.extra_paths,
            disco_threshold=disco_threshold,
            output_tif=tif_path,
        )
        if yr is None:
            logging.warning(f"  Skipping {spec.label} due to load error.")
            continue

        yr.label = spec.label
        _compute_separability(yr)
        year_results.append(yr)

    if not year_results:
        logging.error("No years processed successfully — aborting plots.")
        return []

    # ── Plots ────────────────────────────────────────────────────────────────
    logging.info("\nGenerating comparative plots …")

    plot_boxwhisker_comparison(
        year_results,
        x_label=x_label,
        title=None,
        output_path=str(out / "boxwhisker_comparison.png"),
        max_strip_pts=max_strip_pts,
    )

    plot_violin_comparison(
        year_results,
        x_label=x_label,
        title=None,
        output_path=str(out / "violin_comparison.png"),
        max_strip_pts=max_strip_pts,
    )

    plot_roc_curves(
        year_results,
        title=None,
        output_path=str(out / "roc_curves.png"),
    )

    plot_dsi_histograms(
        year_results,
        x_label=x_label,
        title=None,
        output_path=str(out / "histogram_grid.png"),
    )

    export_summary_csv(year_results, str(out / "separability_summary.csv"))

    logging.info(
        f"\n{'=' * 55}\n"
        f"  Multi-year model application complete.\n"
        f"  Years processed: {[yr.label for yr in year_results]}\n"
        f"  Outputs → {out}\n"
        f"{'=' * 55}"
    )
    return year_results


def apply_dsi_to_single_year(
        pkl_path: str,
        spec: YearSpec,
        output_dir: str,
        disco_threshold: float = 4000.0,
) -> YearResult | None:
    """
    Load a trained model result from pkl_path, apply it to a single YearSpec,
    and save the probability raster. No comparative plots or CSV.

    Parameters
    ----------
    pkl_path        : Path to dsi_result.pkl produced by run_dsi_pipeline().
    spec            : YearSpec for the single year/scenario to process.
    output_dir      : Directory to write the output raster to.
    disco_threshold : Raw discoloration value above which a pixel is "discolored".

    Returns
    -------
    YearResult, or None if loading/application failed.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading trained DSI model from '{pkl_path}' …")
    with open(pkl_path, "rb") as fh:
        dsi_result: dict = pickle.load(fh)

    norm_params: dict = dsi_result["norm_params"]
    model: LogisticRegression = dsi_result["final_model"]

    tif_path = str(out / f"prob_{spec.label}.tif")

    logging.info(f"Processing year/scenario: {spec.label} …")
    yr = _load_and_apply(
        awc_path=spec.awc_path,
        disco_path=spec.disco_path,
        norm_params=norm_params,
        model=model,
        extra_paths=spec.extra_paths,
        disco_threshold=disco_threshold,
        output_tif=tif_path,
    )

    if yr is None:
        logging.warning(f"  Skipping {spec.label} due to load error.")
        return None

    yr.label = spec.label
    logging.info(f"Raster written → {tif_path}")
    return yr

if __name__ == "__main__":
    pass