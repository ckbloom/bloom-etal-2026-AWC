import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

COLOR_C0 = "#005f73"
COLOR_C1 = "#ca6702"
COLOR_HIST = "#ee9b00"


def load_global_stats(path: str) -> list[dict]:
    """Load the pickled records list containing raw class arrays."""
    with open(path, "rb") as f:
        records = pickle.load(f)
    logging.info(f"Loaded {len(records)} records from {path}")
    return records


def compute_quartile_stats(bin_centers: np.ndarray, proportions: np.ndarray,
                           totals: np.ndarray) -> dict | None:
    """
    Computes quartile-based risk statistics across valid AWC bins.

    Bins are divided into Q1 (lowest 25% of AWC) and Q3 (highest 25% of AWC)
    using bin-center percentiles. Discoloration rates are pooled within each
    quartile by summing raw counts.

    Returns a dict with keys:
        q1_awc_range, q2_awc_range, q3_awc_range,
        q1_prop, q2_prop, q3_prop,
        q1_n, q2_n, q3_n,
        rr_q1_vs_q3, rr_q1_vs_q2,
        rr_ci_lo, rr_ci_hi,
        arr_q1_vs_q3
    Returns None if fewer than 4 valid bins or any quartile is empty.
    """
    from scipy import stats as sp_stats

    n = len(bin_centers)
    if n < 4:
        return None

    q25 = np.percentile(bin_centers, 25)
    q75 = np.percentile(bin_centers, 75)

    q1_mask = bin_centers <= q25
    q2_mask = (bin_centers > q25) & (bin_centers <= q75)
    q3_mask = bin_centers > q75

    if not (q1_mask.any() and q2_mask.any() and q3_mask.any()):
        return None

    discolored = np.round(proportions * totals).astype(int)

    def _pool(mask):
        k = discolored[mask].sum()
        n_sites = totals[mask].sum()
        p = k / n_sites if n_sites > 0 else 0.0
        return int(k), int(n_sites), p

    k1, n1, p1 = _pool(q1_mask)
    k2, n2, p2 = _pool(q2_mask)
    k3, n3, p3 = _pool(q3_mask)

    if p3 == 0 or p2 == 0:
        return None

    rr_q1_vs_q3 = p1 / p3
    rr_q1_vs_q2 = p1 / p2

    if k1 > 0 and k3 > 0:
        se_log_rr = np.sqrt(1 / k1 - 1 / n1 + 1 / k3 - 1 / n3)
        log_rr = np.log(rr_q1_vs_q3)
        z = sp_stats.norm.ppf(0.975)
        rr_ci_lo = np.exp(log_rr - z * se_log_rr)
        rr_ci_hi = np.exp(log_rr + z * se_log_rr)
    else:
        rr_ci_lo = rr_ci_hi = np.nan

    return dict(
        q1_awc_range=(bin_centers[q1_mask].min(), bin_centers[q1_mask].max()),
        q2_awc_range=(bin_centers[q2_mask].min(), bin_centers[q2_mask].max()),
        q3_awc_range=(bin_centers[q3_mask].min(), bin_centers[q3_mask].max()),
        q1_prop=p1, q2_prop=p2, q3_prop=p3,
        q1_n=n1, q2_n=n2, q3_n=n3,
        rr_q1_vs_q3=rr_q1_vs_q3,
        rr_q1_vs_q2=rr_q1_vs_q2,
        rr_ci_lo=rr_ci_lo,
        rr_ci_hi=rr_ci_hi,
        arr_q1_vs_q3=p1 - p3,
    )


def plot_mean_proportion_over_dates(
        records: list[dict],
        depth: str,
        n_bins: int = 25,
        min_pixels: int = 50,
        show_quartiles: bool = True,
        output_dir: Path | None = None,
        label_bars: bool = False
):
    """
    Plots the mean proportion of discoloration with SD error bars for each date
    on the EXACT SAME Y-AXIS SCALE using an accessible color palette.

    Optional quartile risk ratio overlay is controlled by `show_quartiles`.
    When enabled, shaded AWC quartile bands (Q1, IQR, Q3) are drawn behind the
    bars, and an annotation box reports pooled proportions, RR Q1 vs Q3,
    RR Q1 vs Q2, and the absolute risk difference — mirroring the per-PTF script.

    NOTE: Because this plot shows means across PTFs, the quartile stats are
    computed from the mean proportions and effective pixel totals (sum across
    PTFs for valid bins). This gives a cross-PTF pooled estimate.
    """
    subset_depth = [r for r in records if r["depth"] == depth]
    if not subset_depth:
        logging.warning(f"No records found for depth={depth}")
        return

    disco_dates = sorted({r["disco_date"] for r in subset_depth})
    datasets = sorted({r["dataset"] for r in subset_depth})

    all_awc = np.concatenate([r["class0"] for r in subset_depth] + [r["class1"] for r in subset_depth])
    global_min, global_max = all_awc.min(), all_awc.max()

    bins = np.linspace(global_min, global_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    plot_data_per_date = {}
    global_max_y = 0.0

    for date in disco_dates:
        ptf_proportions = []
        ptf_totals = []  # track per-PTF totals per bin for pooling into quartile stats

        for dataset in datasets:
            subset = [r for r in subset_depth if r["dataset"] == dataset and r["disco_date"] == date]
            if not subset:
                continue

            c0 = np.concatenate([r["class0"] for r in subset])
            c1 = np.concatenate([r["class1"] for r in subset])
            combined = np.concatenate([c0, c1])

            hist_tot, _ = np.histogram(combined, bins=bins)
            hist_disc, _ = np.histogram(c1, bins=bins)

            prop = np.full_like(hist_tot, np.nan, dtype=float)
            valid = hist_tot >= min_pixels
            prop[valid] = hist_disc[valid] / hist_tot[valid]
            ptf_proportions.append(prop)
            ptf_totals.append(hist_tot)

        if not ptf_proportions:
            continue

        ptf_proportions = np.array(ptf_proportions)  # shape: (n_ptf, n_bins)
        ptf_totals = np.array(ptf_totals)  # shape: (n_ptf, n_bins)

        valid_ptf_count = np.sum(~np.isnan(ptf_proportions), axis=0)
        plot_mask = valid_ptf_count >= 2

        mean_prop = np.full(n_bins, np.nan)
        sd_prop = np.full(n_bins, np.nan)
        mean_prop[plot_mask] = np.nanmean(ptf_proportions[:, plot_mask], axis=0)
        sd_prop[plot_mask] = np.nanstd(ptf_proportions[:, plot_mask], axis=0)

        # Pooled totals for valid bins (sum across PTFs) — used for quartile stats
        pooled_totals = np.sum(ptf_totals, axis=0)

        local_max = np.nanmax(mean_prop + sd_prop) if np.any(~np.isnan(mean_prop + sd_prop)) else 0.0
        global_max_y = max(global_max_y, local_max)

        plot_data_per_date[date] = {
            "mean": mean_prop,
            "sd": sd_prop,
            "mask": plot_mask,
            "pooled_totals": pooled_totals,
        }

    shared_ylim = min(1.0, global_max_y * 1.1) if global_max_y > 0 else 1.0

    for date, data in plot_data_per_date.items():
        fig, ax = plt.subplots(figsize=(9, 6))

        # --- QUARTILE BACKGROUND SHADING ---
        qr = None
        if show_quartiles:
            valid = data["mask"]
            valid_centers = bin_centers[valid]
            valid_means = data["mean"][valid]
            valid_totals = data["pooled_totals"][valid].astype(float)

            # Only compute if enough valid bins and no all-NaN means
            finite_mask = np.isfinite(valid_means)
            if finite_mask.sum() >= 4:
                qr = compute_quartile_stats(
                    bin_centers=valid_centers[finite_mask],
                    proportions=valid_means[finite_mask],
                    totals=valid_totals[finite_mask],
                )

        if qr is not None:
            valid_centers_finite = bin_centers[data["mask"]][np.isfinite(data["mean"][data["mask"]])]
            q25_awc = np.percentile(valid_centers_finite, 25)
            q75_awc = np.percentile(valid_centers_finite, 75)
            x_min, x_max = bins.min(), bins.max()

            ax.axvspan(x_min, q25_awc, color="#cccccc", alpha=0.35, zorder=0)
            ax.axvspan(q25_awc, q75_awc, color="#bbbbbb", alpha=0.0, zorder=0)
            ax.axvspan(q75_awc, x_max, color="#cccccc", alpha=0.35, zorder=0)

            for xv, lbl in [(q25_awc, "Q1/Q2"), (q75_awc, "Q2/Q3")]:
                ax.axvline(xv, color="#333333", linewidth=0.9, linestyle=":", zorder=1)
                # ax.text(xv, 0, f" {lbl}", fontsize=7, color="#333333",
                #         va="bottom", ha="left", rotation=90,
                #         transform=ax.get_xaxis_transform())

        # --- ERROR BAR BARS ---
        bars = ax.bar(
            bin_centers[data["mask"]], data["mean"][data["mask"]], yerr=data["sd"][data["mask"]],
            width=bin_width * 0.85, color=COLOR_HIST, edgecolor="#2b2d42", alpha=0.85, capsize=4,
            error_kw={'elinewidth': 1.5, 'alpha': 0.8, 'ecolor': '#2b2d42'},
            zorder=2
        )

        # --- PROPORTION LABELS ON BARS ---
        if label_bars:
            valid_means = data["mean"][data["mask"]]
            labels = [f"{v:.3f}" for v in valid_means]
            ax.bar_label(
                bars,
                labels=labels,
                label_type='edge',
                padding=5,
                rotation=90,
                fontsize=7.5,
                color="#111111",
                weight='semibold'
            )

        # --- QUARTILE ANNOTATION BOX ---
        # if qr is not None:
        #     ci_str = (f"[{qr['rr_ci_lo']:.2f}–{qr['rr_ci_hi']:.2f}]"
        #               if not np.isnan(qr['rr_ci_lo']) else "[CI n/a]")
        #
        #     q_text = (
        #         f"Quartile risk summary (pooled across PTFs)\n"
        #         f"Q1 AWC {qr['q1_awc_range'][0]:.0f}–{qr['q1_awc_range'][1]:.0f} mm"
        #         f"  p = {qr['q1_prop']:.3f}  (n={qr['q1_n']:,})\n"
        #         f"IQR AWC {qr['q2_awc_range'][0]:.0f}–{qr['q2_awc_range'][1]:.0f} mm"
        #         f"  p = {qr['q2_prop']:.3f}  (n={qr['q2_n']:,})\n"
        #         f"Q3 AWC {qr['q3_awc_range'][0]:.0f}–{qr['q3_awc_range'][1]:.0f} mm"
        #         f"  p = {qr['q3_prop']:.3f}  (n={qr['q3_n']:,})\n"
        #         f"RR Q1 vs Q3 = {qr['rr_q1_vs_q3']:.2f} {ci_str}\n"
        #         f"RR Q1 vs IQR = {qr['rr_q1_vs_q2']:.2f}  |  "
        #         f"ARD = {qr['arr_q1_vs_q3']:+.3f}"
        #     )
        #     ax.text(
        #         0.97, 0.97, q_text,
        #         transform=ax.transAxes,
        #         fontsize=8.0,
        #         verticalalignment='top',
        #         horizontalalignment='right',
        #         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#333333', alpha=0.85),
        #         color="#333333"
        #     )

        # ax.set_title(f"Mean Proportion of Discoloration Across {len(datasets)} PTFs\nDepth: {depth} · Date: {date}",
        #              fontsize=14)
        ax.set_xlabel("AWC (mm)", fontsize=24)
        ax.set_ylabel("Mean Proportion Discolored", fontsize=24)

        ax.tick_params(axis='both', which='major', labelsize=20)
        # ax.set_xlim(bins.min(), bins.max())
        # ax.set_ylim(0, shared_ylim * 1.22 if shared_ylim < 0.8 else shared_ylim * 1.18)
        ax.set_ylim(0, 0.65)

        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()

        if output_dir:
            fig.savefig(os.path.join(output_dir, f"mean_proportion_{depth}_{date}.png"), dpi=300)

        plt.show()

        plt.close(fig)


def plot_mean_kde_comparison_per_date(
        records: list[dict],
        depth: str,
        output_dir: Path | None = None
):
    """
    Generates one plot per date comparing the Mean +/- SD KDE density curves
    of Class 0 vs Class 1 across all PTFs using colorblind-friendly Teal and Orange.
    """
    subset_depth = [r for r in records if r["depth"] == depth]
    if not subset_depth:
        logging.warning(f"No records found for depth={depth}")
        return

    disco_dates = sorted({r["disco_date"] for r in subset_depth})
    datasets = sorted({r["dataset"] for r in subset_depth})

    all_c0 = np.concatenate([r["class0"] for r in subset_depth])
    all_c1 = np.concatenate([r["class1"] for r in subset_depth])
    global_min = min(all_c0.min(), all_c1.min())
    global_max = max(all_c0.max(), all_c1.max())

    padding = (global_max - global_min) * 0.1
    x_eval = np.linspace(global_min - padding, global_max + padding, 300)

    processed_plots = {}
    global_max_density = 0.0

    for date in disco_dates:
        ptf_kdes_c0 = []
        ptf_kdes_c1 = []

        for dataset in datasets:
            subset = [r for r in subset_depth if r["dataset"] == dataset and r["disco_date"] == date]
            if not subset:
                continue

            c0 = np.concatenate([r["class0"] for r in subset])
            c1 = np.concatenate([r["class1"] for r in subset])

            if len(c0) > 1 and np.var(c0) > 1e-6:
                kde_c0 = stats.gaussian_kde(c0)
                ptf_kdes_c0.append(kde_c0(x_eval))

            if len(c1) > 1 and np.var(c1) > 1e-6:
                kde_c1 = stats.gaussian_kde(c1)
                ptf_kdes_c1.append(kde_c1(x_eval))

        if not ptf_kdes_c0 or not ptf_kdes_c1:
            continue

        ptf_kdes_c0 = np.array(ptf_kdes_c0)
        ptf_kdes_c1 = np.array(ptf_kdes_c1)

        mean_c0 = np.mean(ptf_kdes_c0, axis=0)
        sd_c0 = np.std(ptf_kdes_c0, axis=0)

        mean_c1 = np.mean(ptf_kdes_c1, axis=0)
        sd_c1 = np.std(ptf_kdes_c1, axis=0)

        local_max = max(np.max(mean_c0 + sd_c0), np.max(mean_c1 + sd_c1))
        global_max_density = max(global_max_density, local_max)

        processed_plots[date] = {
            "mean_c0": mean_c0, "sd_c0": sd_c0,
            "mean_c1": mean_c1, "sd_c1": sd_c1
        }

    shared_ylim = global_max_density * 1.1 if global_max_density > 0 else 1.0

    for date, data in processed_plots.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(x_eval, data["mean_c0"], label="Not Discolored (Class 0)", color=COLOR_C0, linewidth=2.5)
        lower_c0 = np.clip(data["mean_c0"] - data["sd_c0"], 0, None)
        ax.fill_between(x_eval, lower_c0, data["mean_c0"] + data["sd_c0"], color=COLOR_C0, alpha=0.18)

        ax.plot(x_eval, data["mean_c1"], label="Discolored (Class 1)", color=COLOR_C1, linewidth=2.5)
        lower_c1 = np.clip(data["mean_c1"] - data["sd_c1"], 0, None)
        ax.fill_between(x_eval, lower_c1, data["mean_c1"] + data["sd_c1"], color=COLOR_C1, alpha=0.18)

        ax.set_title(f"AWC Density Comparison Across PTFs\nDepth: {depth} · Date: {date}", fontsize=14)
        ax.set_xlabel("AWC (mm)", fontsize=12)
        ax.set_ylabel("Mean Density", fontsize=12)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(axis='both', linestyle='--', alpha=0.4)
        ax.set_xlim(global_min - padding / 2, global_max + padding / 2)
        ax.set_ylim(0, shared_ylim)

        plt.tight_layout()
        if output_dir:
            out_path = output_dir / f"kde_comparison_{depth}_{date}.png"
            fig.savefig(out_path, dpi=300)
            logging.info(f"Saved -> {out_path}")
        else:
            plt.show()
        plt.close(fig)


def plot_mean_kde_over_time(
        records: list[dict],
        depth: str,
        output_dir: Path | None = None
):
    """Plots the mean KDE curve of class1 with a shaded SD region across PTFs using a clear sequential palette."""
    subset_depth = [r for r in records if r["depth"] == depth]
    if not subset_depth:
        return

    disco_dates = sorted({r["disco_date"] for r in subset_depth})
    datasets = sorted({r["dataset"] for r in subset_depth})

    all_class1 = np.concatenate([r["class1"] for r in subset_depth])
    global_min, global_max = all_class1.min(), all_class1.max()
    padding = (global_max - global_min) * 0.1
    x_eval = np.linspace(global_min - padding, global_max + padding, 300)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Viridis", n_colors=len(disco_dates))

    for date, color in zip(disco_dates, colors):
        date_kdes = []
        for dataset in datasets:
            subset = [r for r in subset_depth if r["dataset"] == dataset and r["disco_date"] == date]
            if not subset:
                continue
            c1 = np.concatenate([r["class1"] for r in subset])
            if len(c1) > 1 and np.var(c1) > 1e-6:
                kde = stats.gaussian_kde(c1)
                date_kdes.append(kde(x_eval))

        if not date_kdes:
            continue

        date_kdes = np.array(date_kdes)
        mean_kde = np.mean(date_kdes, axis=0)
        sd_kde = np.std(date_kdes, axis=0)

        ax.plot(x_eval, mean_kde, label=date, color=color, linewidth=2.5)
        lower_bound = np.clip(mean_kde - sd_kde, 0, None)
        ax.fill_between(x_eval, lower_bound, mean_kde + sd_kde, color=color, alpha=0.12)

    ax.set_title(f"Mean Discolored AWC Density Across {len(datasets)} PTFs Over Time\nDepth: {depth}", fontsize=14)
    ax.set_xlabel("AWC (mm)", fontsize=12)
    ax.set_ylabel("Mean Density", fontsize=12)
    ax.legend(title="Observation Date", loc="upper right")
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    ax.set_xlim(global_min - padding / 2, global_max + padding / 2)

    plt.tight_layout()
    if output_dir:
        fig.savefig(output_dir / f"mean_kde_over_time_{depth}.png", dpi=300)
    else:
        plt.show()
    plt.close(fig)


def print_pixel_totals_by_month(records: list[dict], depth: str | None = None) -> None:
    """
    Prints the total number of discolored (class1) and not-discolored (class0)
    pixels for each observation date, optionally filtered by depth.

    Args:
        records:  The list of dicts loaded from the pickle file.
        depth:    If provided, only counts pixels for that depth (e.g. "mrd", "1m", "2m").
                  If None, all depths are included.
    """
    subset = records if depth is None else [r for r in records if r["depth"] == depth]

    if not subset:
        print(f"No records found for depth='{depth}'.")
        return

    # Aggregate pixel counts per date
    from collections import defaultdict
    totals = defaultdict(lambda: {"discolored": 0, "not_discolored": 0})

    for r in subset:
        date = r["disco_date"]
        totals[date]["not_discolored"] += len(r["class0"])
        totals[date]["discolored"] += len(r["class1"])

    depth_label = depth if depth else "all depths"
    print(f"\nPixel totals by observation date ({depth_label}):")
    print(f"{'Date':<15} {'Not Discolored':>18} {'Discolored':>14} {'Total':>12} {'% Discolored':>14}")
    print("-" * 75)

    for date in sorted(totals):
        nd = totals[date]["not_discolored"]
        d = totals[date]["discolored"]
        tot = nd + d
        pct = 100.0 * d / tot if tot > 0 else 0.0
        print(f"{date:<15} {nd:>18,} {d:>14,} {tot:>12,} {pct:>13.2f}%")

    print()


if __name__ == "__main__":
    pass