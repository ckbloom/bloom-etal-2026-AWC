"""
drought_susceptibility_index.py
================================
Drought Susceptibility Index (DSI) logistic regression pipeline
(spatial-CV-only version)

  1. Standardize AWC (+ optional extra variables): center/scale by mean/std,
     fit fresh inside every outer CV fold's training portion (no leakage),
     and once more on the full dataset for the deployment model.
  2. NESTED spatial cross-validation:
       - OUTER GroupKFold (spatial blocks) -> honest held-out predictions
         for every pixel, pooled + per-fold, used for evaluation. This is
         the analogue of your old checkerboard test set, except every pixel
         gets to play test set at some point.
       - INNER GroupKFold (spatial blocks) inside each outer training fold
         -> RandomizedSearchCV over C, so hyperparameter selection never
         touches the same fold used to evaluate it.
  3. Per outer fold: calibration bins, ROC/PR AUC, Brier (+ skill scores),
     F1/precision/recall, coefficients. Collected across folds for
     mean/std/min/max/CV% summaries (coefficient + performance stability).
  4. Two calibration plots:
       - pooled out-of-fold calibration (1:1-style scatter, like before)
       - fold-level calibration as boxplots-with-fliers per bin, showing
         spread across spatial folds instead of a single point per bin
  5. A final model refit on ALL data (median outer-fold C) for producing
     probability rasters. The *reported* metrics come from the outer CV
     folds, not from this refit model's in-sample fit.
"""

import csv
import pickle
import logging
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    f1_score,
    precision_score,
    average_precision_score,
    recall_score,
    precision_recall_curve,
)
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from scipy.stats import loguniform

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


# ──────────────────────────────────────────────────────────────────────────────
# Spatial block IDs (used for both outer eval CV and inner tuning CV)
# ──────────────────────────────────────────────────────────────────────────────

def make_block_id_grid(da: xr.DataArray, block_size_m: float) -> np.ndarray:
    """Assign each pixel an integer spatial block ID (rows x cols array)."""
    x_coords = da.x.values
    y_coords = da.y.values

    res_x = abs(float(x_coords[1] - x_coords[0]))
    res_y = abs(float(y_coords[1] - y_coords[0]))

    ppt_x = max(1, round(block_size_m / res_x))
    ppt_y = max(1, round(block_size_m / res_y))

    col_tile = np.arange(len(x_coords)) // ppt_x
    row_tile = np.arange(len(y_coords)) // ppt_y
    n_col_tiles = int(col_tile.max()) + 1

    block_id = row_tile[:, np.newaxis].astype(np.int64) * n_col_tiles + col_tile[np.newaxis, :].astype(np.int64)

    logging.info(
        f"Spatial block grid: {ppt_x}x{ppt_y} px/block ({block_size_m / 1000:.0f} km blocks), "
        f"{int(block_id.max()) + 1} unique blocks."
    )
    return block_id


# ──────────────────────────────────────────────────────────────────────────────
# Standardization only
# ──────────────────────────────────────────────────────────────────────────────

def _fit_standard_params(
        arr: np.ndarray,
        name: str,
        unit: str,
        clip_quantiles: tuple[float, float] | None,
) -> dict:
    """Center/scale (mean/std) params for one variable, with optional outlier clipping."""
    if clip_quantiles is not None:
        q_lo = float(np.percentile(arr, clip_quantiles[0] * 100))
        q_hi = float(np.percentile(arr, clip_quantiles[1] * 100))
    else:
        q_lo, q_hi = float("-inf"), float("inf")

    clipped = np.clip(arr, q_lo, q_hi)
    center = float(clipped.mean())
    scale = float(clipped.std())
    if scale == 0.0:
        scale = 1.0

    return {
        "name": name, "unit": unit,
        "center": center, "scale": scale,
        "clip_lower": q_lo, "clip_upper": q_hi,
    }


def _apply_standard(arr: np.ndarray, p: dict) -> np.ndarray:
    clipped = np.clip(arr, p["clip_lower"], p["clip_upper"])
    return (clipped - p["center"]) / p["scale"]


def build_feature_matrix(
        awc_arr: np.ndarray,
        extra_arrs: list[np.ndarray],
        clip_quantiles: tuple[float, float] | None = (0.001, 0.999),
        norm_params: dict | None = None,
        extra_labels: list[str] | None = None,
        extra_units: list[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Build the standardized feature matrix X = [AWC_scaled, extra1_scaled, ...]."""
    n_extras = len(extra_arrs)
    labels = extra_labels or [f"Var{i + 1}" for i in range(n_extras)]
    units = extra_units or ["" for _ in range(n_extras)]

    if norm_params is None:
        awc_params = _fit_standard_params(awc_arr[~np.isnan(awc_arr)], "AWC", "cm", clip_quantiles)
        extra_params = [
            _fit_standard_params(arr[~np.isnan(arr)], labels[i], units[i], clip_quantiles)
            for i, arr in enumerate(extra_arrs)
        ]
        norm_params = {
            "method": "standard",
            "awc": awc_params,
            "extras": extra_params,
            "feature_names": [awc_params["name"]] + [p["name"] for p in extra_params],
        }

    awc_scaled = _apply_standard(awc_arr, norm_params["awc"])
    extra_scaled = [_apply_standard(arr, p) for arr, p in zip(extra_arrs, norm_params["extras"])]

    X = np.column_stack([awc_scaled] + extra_scaled).astype(np.float32)
    return X, norm_params


# ──────────────────────────────────────────────────────────────────────────────
# Predictor multicollinearity diagnostics (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def compute_multicollinearity_diagnostics(X: np.ndarray, feature_names: list[str]) -> dict | None:
    """Pairwise Pearson r/r^2 and VIF per predictor, computed on the fitted design matrix."""
    n_features = X.shape[1]
    if n_features < 2:
        logging.info("  Only one predictor -- skipping pairwise correlation / VIF.")
        return None

    corr_r = np.corrcoef(X, rowvar=False)
    corr_r2 = corr_r ** 2

    pairwise = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            pairwise.append({
                "feature_1": feature_names[i], "feature_2": feature_names[j],
                "pearson_r": float(corr_r[i, j]), "pearson_r2": float(corr_r2[i, j]),
            })

    vif = {}
    for i in range(n_features):
        y_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        reg = LinearRegression().fit(X_others, y_i)
        r2_i = float(reg.score(X_others, y_i))
        vif_i = float("inf") if r2_i >= 1.0 else 1.0 / (1.0 - r2_i)
        vif[feature_names[i]] = {"r2_vs_others": r2_i, "vif": vif_i}

    return {"pairwise": pairwise, "vif": vif}


# ──────────────────────────────────────────────────────────────────────────────
# Calibration binning + skill scores (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def compute_bin_edges(p: np.ndarray, n_bins: int, bin_strategy: Literal["uniform", "quantile"]) -> np.ndarray:
    if bin_strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    elif bin_strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 3:
            logging.warning("  Quantile bin edges collapsed -- falling back to uniform.")
            return np.linspace(0.0, 1.0, n_bins + 1)
        edges[0], edges[-1] = 0.0, 1.0
        return edges
    raise ValueError(f"Unknown bin_strategy: {bin_strategy!r}")


def bin_stats(p: np.ndarray, y: np.ndarray, edges: np.ndarray, min_pixels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign (p, y) into fixed bin edges; return per-bin (mean predicted p, observed proportion, n)."""
    n_bins = len(edges) - 1
    bin_idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    mean_p = np.full(n_bins, np.nan)
    prop = np.full(n_bins, np.nan)
    n = np.zeros(n_bins)
    for b in range(n_bins):
        sel = bin_idx == b
        n[b] = sel.sum()
        if n[b] >= min_pixels:
            mean_p[b] = p[sel].mean()
            prop[b] = y[sel].mean()
    return mean_p, prop, n


def brier_skill_score(brier: float, prevalence: float) -> tuple[float, float]:
    """BSS relative to the naive constant-prevalence baseline."""
    if prevalence <= 0.0 or prevalence >= 1.0:
        return 0.0, float("nan")
    baseline = prevalence * (1.0 - prevalence)
    return baseline, 1.0 - (brier / baseline)


def pr_auc_skill_ratio(pr_auc: float, prevalence: float) -> float:
    """PR AUC as a multiple of the no-skill (prevalence) baseline."""
    if prevalence <= 0.0 or prevalence >= 1.0:
        return float("nan")
    return pr_auc / prevalence


def _format_skill_ratio(ratio: float) -> str:
    if ratio is None or np.isnan(ratio):
        return "n/a (single-class set)"
    return f"{ratio:.2f} x Baseline"


# ──────────────────────────────────────────────────────────────────────────────
# Nested spatial cross-validation (replaces checkerboard split + inner CV)
# ──────────────────────────────────────────────────────────────────────────────

def run_nested_spatial_cv(
        awc_arr: np.ndarray,
        extra_arrs: list[np.ndarray],
        y: np.ndarray,
        groups: np.ndarray,
        feature_names_labels: tuple[list[str], list[str]],
        outer_n_splits: int = 10,
        inner_n_splits: int = 5,
        n_iter: int = 20,
        max_iter: int = 2000,
        threshold_mode: Literal["fixed", "per_fold_f1"] = "per_fold_f1",
        random_state: int = 42,
) -> dict:
    """
    OUTER GroupKFold -> held-out prediction per pixel (evaluation).
    INNER GroupKFold, run separately inside each outer training fold
    -> RandomizedSearchCV over C (hyperparameter tuning never sees the
    outer validation fold).

    Standardization is refit inside each outer fold's training portion
    only, then applied to that fold's validation portion.

    Returns a dict with per-fold rows, pooled out-of-fold predictions,
    and the outer group assignment (for downstream binning/plots).
    """
    extra_labels, extra_units = feature_names_labels
    n = len(y)
    outer_gkf = GroupKFold(n_splits=outer_n_splits)

    oof_p = np.full(n, np.nan)
    fold_rows: list[dict] = []
    fold_raw: list[dict] = []  # per-fold raw (p_val, y_val) for the boxplot calibration plot

    for fold_idx, (tr_idx, val_idx) in enumerate(outer_gkf.split(np.zeros(n), y, groups=groups)):
        X_tr, scale_params = build_feature_matrix(
            awc_arr[tr_idx], [a[tr_idx] for a in extra_arrs],
            norm_params=None, extra_labels=extra_labels, extra_units=extra_units,
        )
        X_val, _ = build_feature_matrix(
            awc_arr[val_idx], [a[val_idx] for a in extra_arrs], norm_params=scale_params,
        )
        y_tr, y_val = y[tr_idx], y[val_idx]
        groups_tr = groups[tr_idx]

        n_inner_groups = len(np.unique(groups_tr))
        this_inner_splits = min(inner_n_splits, n_inner_groups)
        inner_cv = GroupKFold(n_splits=this_inner_splits)

        search = RandomizedSearchCV(
            estimator=LogisticRegression(solver="lbfgs", max_iter=max_iter, class_weight=None),
            param_distributions={"C": loguniform(1e-3, 1e2)},
            n_iter=n_iter, cv=inner_cv, scoring="roc_auc", n_jobs=-1, random_state=random_state,
        )
        search.fit(X_tr, y_tr, groups=groups_tr)
        model = search.best_estimator_
        best_C = float(search.best_params_["C"])

        p_tr = model.predict_proba(X_tr)[:, 1]
        p_val = model.predict_proba(X_val)[:, 1]
        oof_p[val_idx] = p_val

        if threshold_mode == "per_fold_f1":
            precisions, recalls, thresholds = precision_recall_curve(y_tr, p_tr)
            denom = precisions + recalls
            f1s = np.divide(2 * precisions * recalls, denom, out=np.zeros_like(denom), where=denom != 0)
            best_idx = int(np.argmax(f1s))
            threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        else:
            threshold = 0.5

        y_val_bin = (p_val >= threshold).astype(int)

        n_classes = len(np.unique(y_val))
        roc = float(roc_auc_score(y_val, p_val)) if n_classes > 1 else float("nan")
        pr = float(average_precision_score(y_val, p_val)) if n_classes > 1 else float("nan")
        brier = float(brier_score_loss(y_val, p_val))
        prevalence = float(y_val.mean())
        baseline, bss = brier_skill_score(brier, prevalence)
        pr_ratio = pr_auc_skill_ratio(pr, prevalence)

        coef_ = model.coef_.ravel()
        intercept_ = float(model.intercept_.ravel()[0])

        row = {
            "fold": fold_idx,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(val_idx)),
            "n_val_blocks": int(len(np.unique(groups[val_idx]))),
            "tuned_C": best_C,
            "threshold": threshold,
            "val_prevalence": prevalence,
            "intercept": intercept_,
            "val_roc_auc": roc,
            "val_pr_auc": pr,
            "val_pr_auc_skill_ratio": pr_ratio,
            "val_brier": brier,
            "val_brier_skill_score": bss,
            "val_f1": float(f1_score(y_val, y_val_bin, zero_division=0)),
            "val_precision": float(precision_score(y_val, y_val_bin, zero_division=0)),
            "val_recall": float(recall_score(y_val, y_val_bin, zero_division=0)),
        }
        for name, c in zip(scale_params["feature_names"], coef_):
            row[f"coef_{name}"] = float(c)
            row[f"odds_ratio_{name}"] = float(np.exp(c))
        fold_rows.append(row)
        fold_raw.append({"fold": fold_idx, "p_val": p_val, "y_val": y_val})

        logging.info(
            f"  Fold {fold_idx}: C={best_C:.4f}  n_val={len(val_idx):,}  "
            f"ROC AUC={roc:.3f}  PR AUC={pr:.3f}  F1={row['val_f1']:.3f}"
        )

    return {
        "fold_rows": fold_rows,
        "fold_raw": fold_raw,
        "oof_p": oof_p,
        "y": y,
        "groups": groups,
    }


def summarize_cv_folds(fold_rows: list[dict], feature_names: list[str]) -> dict:
    """Mean/std/min/max/CV% across outer folds for coefficients + performance metrics."""
    perf_keys = [
        "val_roc_auc", "val_pr_auc", "val_pr_auc_skill_ratio", "val_brier",
        "val_brier_skill_score", "val_f1", "val_precision", "val_recall",
        "val_prevalence", "tuned_C", "intercept",
    ]
    coef_keys = [f"coef_{name}" for name in feature_names]
    summary = {}
    for k in perf_keys + coef_keys:
        vals = np.array([row[k] for row in fold_rows if k in row], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        mean = float(vals.mean())
        std = float(vals.std())
        summary[k] = {
            "mean": mean, "std": std,
            "min": float(vals.min()), "max": float(vals.max()),
            "cv_pct": float(100.0 * std / abs(mean)) if mean != 0 else float("nan"),
        }
    return summary


def fit_final_model(
        awc_arr: np.ndarray,
        extra_arrs: list[np.ndarray],
        y: np.ndarray,
        fold_rows: list[dict],
        extra_labels: list[str],
        extra_units: list[str],
        max_iter: int = 2000,
) -> tuple[LogisticRegression, dict]:
    """
    Refit on ALL data using the median tuned C across outer folds, for
    deployment (raster prediction). NOT used for the reported performance
    metrics -- those come from the outer CV folds, since this refit sees
    every pixel and its in-sample fit is not a fair generalization estimate.
    """
    median_C = float(np.median([row["tuned_C"] for row in fold_rows]))
    X_all, norm_params = build_feature_matrix(
        awc_arr, extra_arrs, norm_params=None, extra_labels=extra_labels, extra_units=extra_units,
    )
    model = LogisticRegression(solver="lbfgs", max_iter=max_iter, C=median_C, class_weight=None)
    model.fit(X_all, y)
    logging.info(f"Final deployment model refit on all data (median outer-fold C = {median_C:.4f})")
    return model, norm_params


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

SIZE_DOMAIN = (10_000, 100_000)
SIZE_RANGE = (10, 250)
_LIN_ANCHORS = np.round(np.linspace(SIZE_DOMAIN[0], SIZE_DOMAIN[1], 4)).astype(int)


def _n_to_size(n, domain=SIZE_DOMAIN, size_range=SIZE_RANGE):
    lo, hi = domain
    s_lo, s_hi = size_range
    t = np.clip((np.clip(np.asarray(n, float), lo, None) - lo) / (hi - lo), 0, 1)
    return s_lo + t * (s_hi - s_lo)


def _make_size_legend(color: str):
    handles, labels = [], []
    for n in _LIN_ANCHORS:
        h = plt.scatter([], [], s=_n_to_size(n), c=color, edgecolors="black", linewidths=0.6)
        handles.append(h)
        labels.append(f"{n:,}")
    return handles, labels


def plot_pooled_calibration(
        oof_p: np.ndarray,
        y: np.ndarray,
        summary: dict,
        feature_names: list[str],
        n_bins: int = 12,
        min_pixels: int = 50,
        bin_strategy: Literal["uniform", "quantile"] = "uniform",
        title: str = "Pooled Out-of-Fold Calibration (Spatial CV)",
        output_path: str | None = None,
) -> float:
    """
    1:1-style scatter using ALL pixels' out-of-fold predictions pooled
    together. Returns the pooled calibration R^2 (weighted by bin pixel
    count, same definition as the old training-fit R^2) so callers can log
    or export it.
    """
    edges = compute_bin_edges(oof_p, n_bins, bin_strategy)
    mean_p, prop, n = bin_stats(oof_p, y, edges, min_pixels)
    valid = ~np.isnan(prop)
    mean_p, prop, n = mean_p[valid], prop[valid], n[valid]

    weights = n / n.sum()
    weighted_mean_prop = float(np.average(prop, weights=weights))
    ss_res = float(np.sum(weights * (prop - mean_p) ** 2))
    ss_tot = float(np.sum(weights * (prop - weighted_mean_prop) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    prevalence = float(np.mean(y))
    pr_ratio = pr_auc_skill_ratio(summary["val_pr_auc"]["mean"], prevalence)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, zorder=3, label="Perfect calibration")
    ax.scatter(mean_p, prop, s=_n_to_size(n), c="#005f73", edgecolors="black", linewidths=0.6, zorder=5)

    size_handles, size_labels = _make_size_legend(color="#005f73")
    ax.legend(size_handles, size_labels, title="Pixel Count", title_fontsize=10, fontsize=10,
              loc="upper left", bbox_to_anchor=(0.02, 0.75), handletextpad=1, labelspacing=0.75)

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Proportion Discolored", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    annotation_parts = [
        f"R\u00b2: {r2:.2f}",
        f"ROC AUC: {summary['val_roc_auc']['mean']:.2f} \u00b1 {summary['val_roc_auc']['std']:.2f}",
        f"PR AUC: {summary['val_pr_auc']['mean']:.2f} \u00b1 {summary['val_pr_auc']['std']:.2f}  "
        f"{_format_skill_ratio(pr_ratio)}",
        f"F1: {summary['val_f1']['mean']:.2f}   Precision: {summary['val_precision']['mean']:.2f}   "
        f"Recall: {summary['val_recall']['mean']:.2f}",
        "Coefficients (mean \u00b1 std across folds):",
    ]
    for name in feature_names:
        s = summary[f"coef_{name}"]
        annotation_parts.append(f"  {name}: {s['mean']:+.3f} \u00b1 {s['std']:.3f}")
    ax.text(0.03, 0.97, "\n".join(annotation_parts), transform=ax.transAxes, fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#333333", alpha=0.85))

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
        logging.info(f"Pooled calibration plot saved -> {output_path}")
    else:
        plt.show()
    plt.close(fig)

    return r2


def plot_fold_calibration_boxplot(
        fold_raw: list[dict],
        oof_p: np.ndarray,
        y: np.ndarray,
        n_bins: int = 12,
        min_pixels: int = 30,
        bin_strategy: Literal["uniform", "quantile"] = "uniform",
        title: str = "Calibration Across Spatial CV Folds",
        output_path: str | None = None,
) -> None:
    """
    Per-bin boxplot (median/IQR/whiskers/fliers) of each fold's observed
    proportion, using ONE fixed set of bin edges (from pooled out-of-fold
    predictions) so bins line up across folds. The pooled proportion is
    overlaid as a diamond, matching the look of the old 1:1 scatter plot.
    """
    edges = compute_bin_edges(oof_p, n_bins, bin_strategy)
    bin_width = np.diff(edges)
    bin_centers_nominal = edges[:-1] + bin_width / 2

    n_fold_bins = len(edges) - 1
    per_bin_values: list[list[float]] = [[] for _ in range(n_fold_bins)]

    for f in fold_raw:
        mean_p, prop, n = bin_stats(f["p_val"], f["y_val"], edges, min_pixels)
        for b in range(n_fold_bins):
            if not np.isnan(prop[b]):
                per_bin_values[b].append(prop[b])

    keep = [b for b in range(n_fold_bins) if len(per_bin_values[b]) >= 2]
    if not keep:
        logging.warning("  Not enough fold coverage per bin to draw the fold-calibration boxplot -- skipping.")
        return

    positions = bin_centers_nominal[keep]
    data = [per_bin_values[b] for b in keep]

    pooled_mean_p, pooled_prop, pooled_n = bin_stats(oof_p, y, edges, min_pixels)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, zorder=2, label="Perfect calibration")

    box_width = float(np.median(bin_width)) * 0.6
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=True,
        manage_ticks=False,
        zorder=4,
        flierprops=dict(
            marker="o",
            markersize=3.5,
            markerfacecolor="#64748b",
            markeredgecolor="none",
            alpha=0.5,
        ),
        boxprops=dict(
            facecolor="#e2e8f0",
            edgecolor="#334155",
            linewidth=1.1,
            alpha=0.70,
        ),
        medianprops=dict(
            color="#64748b",
            linewidth=1.5,
            linestyle="-",
        ),
        whiskerprops=dict(color="#334155", linewidth=1.1),
        capprops=dict(color="#334155", linewidth=1.1),
    )

    # Mean/Pooled Overlay Diamonds (Navy Sapphire)
    pooled_valid = ~np.isnan(pooled_prop)
    ax.scatter(
        pooled_mean_p[pooled_valid],
        pooled_prop[pooled_valid],
        marker="D",
        s=42,
        c="#1e3a8a",
        edgecolors="#ffffff",
        linewidths=0.9,
        zorder=6,
    )

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Proportion Discolored", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
        logging.info(f"Fold calibration boxplot saved -> {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_coefficient_stability(
        fold_rows: list[dict],
        feature_names: list[str],
        title: str = "Coefficient Stability Across Outer Spatial CV Folds",
        output_path: str | None = None,
) -> None:
    keys = ["intercept"] + [f"coef_{name}" for name in feature_names]
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 5), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, k in zip(axes, keys):
        vals = np.array([row[k] for row in fold_rows], dtype=float)
        folds = [row["fold"] for row in fold_rows]
        mean, std = float(vals.mean()), float(vals.std())

        jitter = (np.arange(len(vals)) - (len(vals) - 1) / 2.0) * 0.06
        ax.scatter(jitter, vals, color="#666666", zorder=3, s=40, edgecolors="black", linewidths=0.5)
        for x, y_, f in zip(jitter, vals, folds):
            ax.annotate(str(f), (x, y_), textcoords="offset points", xytext=(4, 2), fontsize=7, color="#666666")

        ax.errorbar([0], [mean], yerr=[std], fmt="D", color="#ca6702", capsize=6, markersize=7, zorder=5,
                    label="mean \u00b1 std")
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_title(k, fontsize=11)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Fold-level value", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, fontsize=9)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    if output_path:
        fig.savefig(output_path, dpi=300)
        logging.info(f"Coefficient stability plot saved -> {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_dsi_pipeline(
        awc_path: str,
        disco_path: str,
        output_dir: str,
        extra_paths: list[str] | None = None,
        extra_labels: list[str] | None = None,
        extra_units: list[str] | None = None,
        cv_block_size_m: float = 30_000.0,
        outer_n_splits: int = 10,
        inner_n_splits: int = 5,
        n_bins: int = 12,
        min_pixels: int = 50,
        disco_threshold: float = 4000.0,
        bin_strategy: Literal["uniform", "quantile"] = "uniform",
        save_pkl: bool = True,
) -> dict:
    plt.style.use("default")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    extra_paths = extra_paths or []
    extra_labels = extra_labels or [f"Var{i + 1}" for i in range(len(extra_paths))]
    extra_units = extra_units or ["" for _ in extra_paths]

    # ── Load rasters ─────────────────────────────────────────────────────────
    logging.info("Loading rasters")
    awc = rioxarray.open_rasterio(awc_path, masked=True)
    disco = rioxarray.open_rasterio(disco_path, masked=True)
    disco_r = disco.rio.reproject_match(awc, resampling=Resampling.average)

    awc_arr_full = awc.values.astype(np.float32).squeeze()
    disco_arr_full = disco_r.values.astype(np.float32).squeeze()

    extra_arrs_full: list[np.ndarray] = []
    for path in extra_paths:
        da = rioxarray.open_rasterio(path, masked=True)
        da_r = da.rio.reproject_match(awc, resampling=Resampling.bilinear)
        extra_arrs_full.append(da_r.values.astype(np.float32).squeeze())

    disco_bin_full = (disco_arr_full >= disco_threshold).astype(np.uint8)

    valid_mask = ~np.isnan(awc_arr_full) & ~np.isnan(disco_arr_full)
    for arr in extra_arrs_full:
        valid_mask &= ~np.isnan(arr)
    logging.info(f"Valid pixels: {valid_mask.sum():,}")

    block_grid = make_block_id_grid(awc, block_size_m=cv_block_size_m)
    groups = block_grid[valid_mask]

    awc_v = awc_arr_full[valid_mask]
    extra_v = [a[valid_mask] for a in extra_arrs_full]
    y_v = disco_bin_full[valid_mask].astype(int)

    # ── Multicollinearity (fit once on full standardized data, diagnostic only) ─
    X_all_diag, norm_params_diag = build_feature_matrix(
        awc_v, extra_v, norm_params=None, extra_labels=extra_labels, extra_units=extra_units,
    )
    logging.info("Computing predictor multicollinearity diagnostics")
    multicollinearity = compute_multicollinearity_diagnostics(X_all_diag, norm_params_diag["feature_names"])

    # ── Nested spatial CV ──────────────────────────────────────────────────────
    logging.info(f"Running nested spatial CV (outer={outer_n_splits} folds, inner={inner_n_splits} folds)")
    cv_result = run_nested_spatial_cv(
        awc_v, extra_v, y_v, groups,
        feature_names_labels=(extra_labels, extra_units),
        outer_n_splits=outer_n_splits, inner_n_splits=inner_n_splits,
    )
    fold_rows = cv_result["fold_rows"]
    fold_raw = cv_result["fold_raw"]
    oof_p = cv_result["oof_p"]

    feature_names = norm_params_diag["feature_names"]
    cv_summary = summarize_cv_folds(fold_rows, feature_names)

    for k, s in cv_summary.items():
        logging.info(f"  {k}: mean={s['mean']:.4f}  std={s['std']:.4f}  CV%={s['cv_pct']:.1f}")

    # ── Final deployment model (refit on all data) ────────────────────────────
    final_model, norm_params = fit_final_model(
        awc_v, extra_v, y_v, fold_rows, extra_labels, extra_units,
    )

    # ── Plots ───────────────────────────────────────────────────────────────
    pooled_calibration_r2 = plot_pooled_calibration(
        oof_p, y_v, cv_summary, feature_names, n_bins=n_bins, min_pixels=min_pixels, bin_strategy=bin_strategy,
        output_path=str(out / "pooled_oof_calibration.png"),
    )
    plot_fold_calibration_boxplot(
        fold_raw, oof_p, y_v, n_bins=n_bins, min_pixels=max(10, min_pixels // 3), bin_strategy=bin_strategy,
        output_path=str(out / "fold_calibration_boxplot.png"),
    )
    plot_coefficient_stability(
        fold_rows, feature_names, output_path=str(out / "coefficient_stability_cv.png"),
    )

    # ── Assemble result ────────────────────────────────────────────────────
    result = {
        "norm_params": norm_params,
        "final_model": final_model,
        "feature_names": feature_names,
        "fold_rows": fold_rows,
        "cv_summary": cv_summary,
        "pooled_calibration_r2": pooled_calibration_r2,
        "oof_p": oof_p,
        "y": y_v,
        "multicollinearity": multicollinearity,
        "run_config": {
            "cv_block_size_m": cv_block_size_m,
            "outer_n_splits": outer_n_splits,
            "inner_n_splits": inner_n_splits,
            "bin_strategy": bin_strategy,
            "n_bins_requested": n_bins,
            "min_pixels": min_pixels,
            "disco_threshold": disco_threshold,
        },
    }

    # ── Export per-fold CSV ────────────────────────────────────────────────
    fold_csv_path = out / "spatial_cv_folds.csv"
    with open(fold_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        for row in fold_rows:
            writer.writerow(row)
    logging.info(f"Per-fold CV metrics written -> {fold_csv_path}")

    # ── Export summary CSV ────────────────────────────────────────────────
    csv_path = out / "dsi_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter / Evaluation Metric", "Value"])
        writer.writerow(["model", "logistic_regression"])
        writer.writerow(["scaling_method", "standard"])
        writer.writerow(["cv_block_size_m", cv_block_size_m])
        writer.writerow(["outer_n_splits", outer_n_splits])
        writer.writerow(["inner_n_splits", inner_n_splits])
        writer.writerow(["bin_strategy", bin_strategy])
        writer.writerow(["n_bins_requested", n_bins])
        writer.writerow(["min_pixels", min_pixels])
        writer.writerow(["disco_threshold", disco_threshold])
        writer.writerow(["pooled_oof_calibration_r2", f"{pooled_calibration_r2:.6f}"])
        writer.writerow(["final_model_C", final_model.C])
        writer.writerow(["final_model_intercept", f"{float(final_model.intercept_[0]):.6f}"])
        for name, coef in zip(feature_names, final_model.coef_.ravel()):
            writer.writerow([f"final_model_coef_{name}", f"{coef:.6f}"])
            writer.writerow([f"final_model_odds_ratio_{name}", f"{np.exp(coef):.6f}"])

        if multicollinearity is not None:
            for p in multicollinearity["pairwise"]:
                pair_label = f"{p['feature_1']}_vs_{p['feature_2']}"
                writer.writerow([f"pearson_r_{pair_label}", f"{p['pearson_r']:.6f}"])
                writer.writerow([f"pearson_r2_{pair_label}", f"{p['pearson_r2']:.6f}"])
            for name, v in multicollinearity["vif"].items():
                writer.writerow([f"vif_{name}", f"{v['vif']:.6f}"])

        for k, s in cv_summary.items():
            writer.writerow([f"cv_{k}_mean", f"{s['mean']:.6f}"])
            writer.writerow([f"cv_{k}_std", f"{s['std']:.6f}"])
            writer.writerow([f"cv_{k}_min", f"{s['min']:.6f}"])
            writer.writerow([f"cv_{k}_max", f"{s['max']:.6f}"])
            writer.writerow([f"cv_{k}_cv_pct", f"{s['cv_pct']:.4f}"])
    logging.info(f"Metrics written -> {csv_path}")

    if save_pkl:
        pkl_path = out / "dsi_result.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(result, fh)
        logging.info(f"DSIResult pickled -> {pkl_path}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Apply the final (all-data) model to a new raster extent
# ──────────────────────────────────────────────────────────────────────────────

def predict_probability_raster(
        awc_path: str,
        norm_params: dict,
        final_model: LogisticRegression,
        extra_paths: list[str] | None = None,
        output_path: str = "discoloration_probability.tif",
) -> xr.DataArray:
    awc = rioxarray.open_rasterio(awc_path, masked=True)
    awc_arr = awc.values.astype(np.float32)

    extra_arrs = []
    for path in (extra_paths or []):
        da = rioxarray.open_rasterio(path, masked=True)
        da_r = da.rio.reproject_match(awc, resampling=Resampling.bilinear)
        extra_arrs.append(da_r.values.astype(np.float32))

    orig_shape = awc_arr.shape
    X, _ = build_feature_matrix(awc_arr.ravel(), [a.ravel() for a in extra_arrs], norm_params=norm_params)
    valid = ~np.isnan(X).any(axis=1)

    prob_flat = np.full(X.shape[0], np.nan, dtype=np.float32)
    prob_flat[valid] = final_model.predict_proba(X[valid])[:, 1].astype(np.float32)
    prob = prob_flat.reshape(orig_shape)

    prob_da = xr.DataArray(prob, dims=awc.dims, coords=awc.coords)
    prob_da.rio.write_crs(awc.rio.crs, inplace=True)
    prob_da = prob_da.fillna(-9999.0)
    prob_da.rio.write_nodata(-9999.0, inplace=True)
    prob_da.name = "discoloration_probability"
    prob_da.rio.to_raster(output_path, compress="lzw")
    logging.info(f"Probability raster written -> {output_path}")
    return prob_da


# ──────────────────────────────────────────────────────────────────────────────
# Batch-run aggregation
# ──────────────────────────────────────────────────────────────────────────────

def summarize_run(ptf: str, depth: str, month: int, result: dict) -> dict:
    cfg = result["run_config"]
    cv = result["cv_summary"]

    row = {
        "ptf": ptf, "depth": depth, "month": month,
        "scaling_method": "standard",
        "cv_block_size_m": cfg["cv_block_size_m"],
        "outer_n_splits": cfg["outer_n_splits"],
        "inner_n_splits": cfg["inner_n_splits"],
        "bin_strategy": cfg["bin_strategy"],
        "n_bins_requested": cfg["n_bins_requested"],
        "min_pixels": cfg["min_pixels"],
        "disco_threshold": cfg["disco_threshold"],
        "pooled_oof_calibration_r2": result["pooled_calibration_r2"],
        "final_model_C": result["final_model"].C,
        "final_model_intercept": float(result["final_model"].intercept_[0]),
    }
    for name, coef in zip(result["feature_names"], result["final_model"].coef_.ravel()):
        row[f"final_model_coef_{name}"] = coef
        row[f"final_model_odds_ratio_{name}"] = float(np.exp(coef))

    for k, s in cv.items():
        row[f"cv_{k}_mean"] = s["mean"]
        row[f"cv_{k}_std"] = s["std"]
        row[f"cv_{k}_cv_pct"] = s["cv_pct"]

    multicollinearity = result.get("multicollinearity")
    if multicollinearity is not None:
        for p in multicollinearity["pairwise"]:
            pair_label = f"{p['feature_1']}_vs_{p['feature_2']}"
            row[f"pearson_r_{pair_label}"] = p["pearson_r"]
        for name, v in multicollinearity["vif"].items():
            row[f"vif_{name}"] = v["vif"]

    return row


def write_aggregate_summary(summary_rows: list[dict], output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not summary_rows:
        logging.warning("No summary rows to write -- skipping aggregate summary.")
        return out / "aggregate_summary.csv"

    fieldnames: list[str] = []
    for row in summary_rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    csv_path = out / "aggregate_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    logging.info(f"Aggregate summary written -> {csv_path} ({len(summary_rows)} runs)")

    pkl_path = out / "aggregate_summary.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(summary_rows, fh)
    logging.info(f"Aggregate summary pickled -> {pkl_path}")

    return csv_path


if __name__ == "__main__":
    pass
