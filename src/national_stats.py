import os
import xarray as xr
import logging
import warnings
import numpy as np
import rioxarray
from scipy import stats
from rasterio.enums import Resampling
import pandas as pd
import pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*angle from rectified to skew grid parameter lost.*",
    module="pyproj"
)


def save_stats(records: list[dict], output_path: str) -> None:
    """Pickle the full records list (including raw class0/class1 arrays)."""
    with open(output_path, "wb") as f:
        pickle.dump(records, f)
    logging.info(f"Stats saved → {output_path}")


def load_stats(path: str) -> list[dict]:
    """Load a pickled records list."""
    with open(path, "rb") as f:
        records = pickle.load(f)
    logging.info(f"Stats loaded ← {path} ({len(records)} records)")
    return records


def update_stats(
        existing: list[dict],
        new_datasets: list[str] | None = None,
        new_depths: list[str] | None = None,
        new_disco_dates: list[str] | None = None,
        awc_input: str = "",
        disco_input: str = "",
        output: str = "",
        disco_threshold: float = 4000,
        stats_path: str = "",
) -> list[dict]:
    """
    Compute stats only for combinations not already present in existing records,
    then merge and return the full updated list.
    """
    # ── Infer full sets from existing + any new additions ─────────────────────
    existing_datasets = {r["dataset"] for r in existing}
    existing_depths = {r["depth"] for r in existing}
    existing_dates = {r["disco_date"] for r in existing}

    all_datasets = existing_datasets | set(new_datasets or [])
    all_depths = existing_depths | set(new_depths or [])
    all_dates = existing_dates | set(new_disco_dates or [])

    # ── Index existing records for O(1) lookup ────────────────────────────────
    existing_keys = {
        (r["dataset"], r["depth"], r["disco_date"])
        for r in existing
    }

    # ── Find missing combinations ─────────────────────────────────────────────
    missing = [
        (dataset, depth, date)
        for dataset in all_datasets
        for depth in all_depths
        for date in all_dates
        if (dataset, depth, date) not in existing_keys
    ]

    if not missing:
        logging.info("No new combinations to compute — cache is up to date.")
        return existing

    logging.info(f"Computing {len(missing)} new combinations...")

    # ── Compute only what's missing ───────────────────────────────────────────
    new_records: list[dict] = []

    for dataset, depth, date in missing:
        mineral_awc = os.path.join(awc_input, f'awc_{depth}_{dataset}.tif')

        # Pointing to the unclipped mosaic file
        disco = os.path.join(disco_input, f"Disco_CH_{date}.tif")

        c0, c1 = prepare_threshold_data(disco, mineral_awc, threshold=disco_threshold)

        run_stats = compute_stats(c0, c1)
        run_stats.update({
            "dataset": dataset, "depth": depth,
            "disco_date": date,
            "class0": c0, "class1": c1,
        })
        new_records.append(run_stats)
        logging.info(f"Computed {dataset} {depth} {date}")

    # ── Merge, persist, return ────────────────────────────────────────────────
    updated = existing + new_records

    # Strip raw arrays before saving to the final summary CSV
    csv_records = [{k: v for k, v in r.items() if k not in ("class0", "class1")}
                   for r in updated]

    save_summary_table(
        csv_records,
        output_path=os.path.join(output, "summary_stats_all.csv"),
    )

    save_stats(updated, stats_path)
    logging.info(f"Cache updated: {len(existing)} → {len(updated)} records.")
    return updated


def load_raster(path: str) -> xr.DataArray:
    """Open a GeoTIFF as a 2-D xarray.DataArray."""
    da = rioxarray.open_rasterio(path, masked=True)
    return da


def prepare_threshold_data(prob_path: str, depth_awc_path: str, threshold: float = 0.5):
    """Loads rasters and returns split depth arrays for two classes."""
    # Load and match
    prob = load_raster(prob_path)
    depth_awc = load_raster(depth_awc_path)

    prob_matched = prob.rio.reproject_match(depth_awc, resampling=Resampling.average)

    # Extract arrays
    prob_arr = prob_matched.values.astype(np.float32)
    depth_arr = depth_awc.values.astype(np.float32)

    # Valid-pixel mask
    valid = ~np.isnan(prob_arr) & ~np.isnan(depth_arr)

    # Binary classification
    binary = (prob_arr >= threshold).astype(np.uint8)
    class0 = depth_arr[(binary == 0) & valid]
    class1 = depth_arr[(binary == 1) & valid]

    return class0, class1


def compute_stats(class0: np.ndarray, class1: np.ndarray) -> dict:
    """Compute a comprehensive set of statistics for two classes."""
    n0, n1 = len(class0), len(class1)

    # Basic location / spread
    median0, median1 = float(np.median(class0)), float(np.median(class1))
    mean0, mean1 = float(np.mean(class0)), float(np.mean(class1))
    iqr0 = float(np.percentile(class0, 75) - np.percentile(class0, 25))
    iqr1 = float(np.percentile(class1, 75) - np.percentile(class1, 25))
    median_diff = median1 - median0
    median_ratio = (median1 / median0) if median0 != 0 else np.nan

    # Combined stats & Overlap Coefficient via shared histogram bins
    combined = np.concatenate([class0, class1])
    mean_awc = float(np.mean(combined))
    median_awc = float(np.median(combined))

    # Mann-Whitney U + rank-biserial r
    u_stat, p_value = stats.mannwhitneyu(class0, class1, alternative="two-sided")
    r_rb = 1 - (2 * u_stat) / (n0 * n1)

    # Cohen's d (pooled SD)
    pooled_sd = np.sqrt(
        ((n0 - 1) * np.var(class0, ddof=1) + (n1 - 1) * np.var(class1, ddof=1))
        / (n0 + n1 - 2)
    )
    cohens_d = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else np.nan

    # Common Language Effect Size: P(X1 > X0)
    cles = u_stat / (n0 * n1)

    # Overlap Coefficient via shared histogram bins
    bins = np.linspace(combined.min(), combined.max(), 200)
    hist0, _ = np.histogram(class0, bins=bins, density=True)
    hist1, _ = np.histogram(class1, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    overlap_coeff = float(np.sum(np.minimum(hist0, hist1) * bin_width))

    return {
        "n0": n0,
        "n1": n1,
        "median0": round(median0, 4),
        "median1": round(median1, 4),
        "mean0": round(mean0, 4),
        "mean1": round(mean1, 4),
        "iqr0": round(iqr0, 4),
        "iqr1": round(iqr1, 4),
        "median_diff": round(median_diff, 4),
        "median_ratio": round(median_ratio, 4) if not np.isnan(median_ratio) else np.nan,
        "u_stat": u_stat,
        "p_value": p_value,
        "r_rb": round(r_rb, 4),
        "cohens_d": round(cohens_d, 4) if not np.isnan(cohens_d) else np.nan,
        "cles": round(cles, 4),
        "overlap_coeff": round(overlap_coeff, 4),
        "mean_awc": round(mean_awc, 4),
        "median_awc": round(median_awc, 4)
    }


def save_summary_table(records: list[dict], output_path: str) -> pd.DataFrame:
    """Convert a list of per-run stat dicts into a tidy DataFrame and save as CSV."""
    df = pd.DataFrame(records)

    # Reorder columns for readability (district removed)
    id_cols = ["dataset", "depth", "disco_date"]
    stat_cols = [
        "n0", "n1",
        "median0", "median1", "mean0", "mean1",
        "iqr0", "iqr1",
        "median_diff", "median_ratio",
        "u_stat", "p_value",
        "r_rb", "cohens_d", "cles", "overlap_coeff", "mean_awc", "median_awc"
    ]
    df = df[id_cols + stat_cols]
    df = df.sort_values("r_rb", key=abs, ascending=False).reset_index(drop=True)

    df.to_csv(output_path, index=False, float_format="%.6g")
    logging.info(f"Summary table saved → {output_path}")
    return df


if __name__ == "__main__":
    pass