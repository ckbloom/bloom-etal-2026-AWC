import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path
import logging
import os
import re
import matplotlib.image as mpimg
import math


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_global_stats(path: str) -> list[dict]:
    """Load the pickled records list containing raw class arrays."""
    with open(path, "rb") as f:
        records = pickle.load(f)
    logging.info(f"Loaded {len(records)} records from {path}")
    return records


def process_depth_scales(records: list[dict], depth: str, datasets: list[str], dates: list[str], n_bins: int,
                         min_pixels: int):
    """
    Pre-scans the data for a given depth to establish a uniform X-axis (bins)
    and find the absolute maximum Y-value (proportion) across all dates and datasets.
    """
    subset_depth = [r for r in records if r["depth"] == depth]
    if not subset_depth:
        logging.warning(f"  --> No records found matching depth='{depth}' at all.")
        return None, 1.0

    matched_records = [r for r in subset_depth if r["dataset"] in datasets and r["disco_date"] in dates]
    logging.info(f"  --> Depth '{depth}': Found {len(matched_records)} records matching your dataset/date filters.")

    if not matched_records:
        logging.warning(
            f"  --> Skipping plotting for '{depth}' because 0 records matched your exact dataset/date text lists.")
        return None, 1.0

    all_c0 = np.concatenate([r["class0"] for r in matched_records])
    all_c1 = np.concatenate([r["class1"] for r in matched_records])
    combined_global = np.concatenate([all_c0, all_c1])

    global_bins = np.linspace(combined_global.min(), combined_global.max(), n_bins + 1)

    global_max_y = 0.0
    for date in dates:
        for ds in datasets:
            subset = [r for r in subset_depth if r["dataset"] == ds and r["disco_date"] == date]
            if not subset:
                continue

            c0 = np.concatenate([r["class0"] for r in subset])
            c1 = np.concatenate([r["class1"] for r in subset])
            comb = np.concatenate([c0, c1])

            hist_total, _ = np.histogram(comb, bins=global_bins)
            hist_disc, _ = np.histogram(c1, bins=global_bins)

            valid = hist_total >= min_pixels
            if np.any(valid):
                props = hist_disc[valid] / hist_total[valid]
                global_max_y = max(global_max_y, props.max())

    shared_ylim = min(1.0, global_max_y * 1.1) if global_max_y > 0 else 1.0
    return global_bins, shared_ylim


def fit_weighted_logit_slope(bin_centers: np.ndarray, proportions: np.ndarray,
                              totals: np.ndarray) -> dict | None:
    """
    Fits a weighted least-squares regression of logit(proportion) ~ AWC bin midpoint.

    Weights are n_i * p_tilde_i * (1 - p_tilde_i), i.e. the inverse of the
    asymptotic variance of logit(p). A continuity correction is applied so that
    bins with p=0 or p=1 don't blow up the logit.

    Returns a dict with keys:
        slope, intercept, slope_se, slope_ci_lo, slope_ci_hi,
        p_value, or_per_unit, or_per_50, fitted_proportions, x_fit
    Returns None if there are fewer than 3 valid bins.
    """
    from scipy import stats as sp_stats

    if len(bin_centers) < 3:
        return None

    # Continuity-corrected proportion to avoid logit blowup at 0/1
    p_tilde = (proportions * totals + 0.5) / (totals + 1.0)

    logit_p = np.log(p_tilde / (1.0 - p_tilde))

    # Variance-stabilising weights: n * p * (1-p)
    weights = totals * p_tilde * (1.0 - p_tilde)

    # Weighted least squares via polyfit (degree 1)
    # np.polyfit with w argument minimises sum(w * residual^2)
    coeffs, cov = np.polyfit(bin_centers, logit_p, deg=1, w=np.sqrt(weights), cov=True)
    slope, intercept = coeffs

    slope_se = np.sqrt(cov[0, 0])
    n_bins = len(bin_centers)
    df = n_bins - 2
    t_stat = slope / slope_se
    p_value = 2.0 * sp_stats.t.sf(np.abs(t_stat), df=df)
    t_crit = sp_stats.t.ppf(0.975, df=df)
    slope_ci_lo = slope - t_crit * slope_se
    slope_ci_hi = slope + t_crit * slope_se

    # Dense x range for smooth fitted curve
    x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 300)
    logit_fit = slope * x_fit + intercept
    fitted_proportions = 1.0 / (1.0 + np.exp(-logit_fit))   # inverse logit → probability

    return dict(
        slope=slope,
        intercept=intercept,
        slope_se=slope_se,
        slope_ci_lo=slope_ci_lo,
        slope_ci_hi=slope_ci_hi,
        p_value=p_value,
        or_per_unit=np.exp(slope),
        or_per_50=np.exp(50 * slope),
        x_fit=x_fit,
        fitted_proportions=fitted_proportions,
    )


def compute_quartile_stats(bin_centers: np.ndarray, proportions: np.ndarray,
                           totals: np.ndarray) -> dict | None:
    """
    Computes quartile-based risk statistics across valid AWC bins.

    Bins are divided into Q1 (lowest 25% of AWC) and Q3 (highest 25% of AWC)
    using bin-center percentiles. Discoloration rates are pooled within each
    quartile by summing raw counts, giving a single site-weighted proportion
    that is not sensitive to bin width or unequal bin sizes.

    Also computes Q1 vs Q2 (median quartile) to specifically test whether the
    lowest AWC bins are elevated relative to typical mid-range AWC sites — the
    ecologically meaningful comparison given that most sites are mid-range.

    Returns a dict with keys:
        q1_awc_range, q3_awc_range,            -- AWC ranges (mm)
        q1_prop, q2_prop, q3_prop,             -- pooled proportions per quartile
        q1_n, q2_n, q3_n,                      -- total site counts per quartile
        rr_q1_vs_q3,                           -- risk ratio: Q1 discoloration / Q3 discoloration
        rr_q1_vs_q2,                           -- risk ratio: Q1 / Q2 (low vs median)
        rr_ci_lo, rr_ci_hi,                    -- 95% CI on Q1 vs Q3 RR (log-normal method)
        arr_q1_vs_q3,                          -- absolute risk reduction Q1 - Q3
    Returns None if fewer than 4 valid bins or any quartile is empty.
    """
    from scipy import stats as sp_stats

    n = len(bin_centers)
    if n < 4:
        return None

    # Quartile boundaries by bin-center AWC value (not by count)
    q25 = np.percentile(bin_centers, 25)
    q50 = np.percentile(bin_centers, 50)
    q75 = np.percentile(bin_centers, 75)

    q1_mask = bin_centers <= q25
    q2_mask = (bin_centers > q25) & (bin_centers <= q75)   # interquartile = "median" zone
    q3_mask = bin_centers > q75

    if not (q1_mask.any() and q2_mask.any() and q3_mask.any()):
        return None

    # Reconstruct raw discolored counts from proportions × totals
    discolored = np.round(proportions * totals).astype(int)

    def _pool(mask):
        k = discolored[mask].sum()
        n_sites = totals[mask].sum()
        p = k / n_sites if n_sites > 0 else 0.0
        return int(k), int(n_sites), p

    k1, n1, p1 = _pool(q1_mask)
    k2, n2, p2 = _pool(q2_mask)
    k3, n3, p3 = _pool(q3_mask)

    # Guard against zero denominators
    if p3 == 0 or p2 == 0:
        return None

    rr_q1_vs_q3 = p1 / p3
    rr_q1_vs_q2 = p1 / p2

    # 95% CI on Q1 vs Q3 RR using log-normal method (Katz et al.)
    # SE(log RR) = sqrt(1/k1 - 1/n1 + 1/k3 - 1/n3)
    if k1 > 0 and k3 > 0:
        se_log_rr = np.sqrt(1/k1 - 1/n1 + 1/k3 - 1/n3)
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


def plot_proportion_histogram_standardized(
        records: list[dict],
        dataset: str,
        depth: str,
        disco_date: str,
        bins: np.ndarray,
        ylim: float,
        min_pixels: int = 100,
        show_regression: bool = True,
        show_quartiles: bool = True,
        vertical_fraction_labels=False,
        output_path: str | None = None
):
    """
    Plots a proportion histogram using pre-standardized global bins and Y-axis limits.
    Labels each bar vertically with the contextual fraction format: discolored / total.

    Optional weighted logit regression line overlay is controlled by `show_regression`.
    When enabled, the fitted probability curve and a statistics annotation are added,
    and the legend reports slope, OR per 50 mm AWC, and p-value.

    Optional quartile risk ratio overlay is controlled by `show_quartiles`.
    When enabled, shaded AWC quartile bands (Q1, IQR, Q3) are drawn behind the bars,
    and a second annotation box reports pooled proportions, RR Q1 vs Q3, RR Q1 vs Q2,
    and the absolute risk difference.
    """
    subset = [r for r in records if r["dataset"] == dataset and r["depth"] == depth and r["disco_date"] == disco_date]
    if not subset:
        return

    class0_all = np.concatenate([r["class0"] for r in subset])
    class1_all = np.concatenate([r["class1"] for r in subset])
    combined = np.concatenate([class0_all, class1_all])

    hist_total, _ = np.histogram(combined, bins=bins)
    hist_discolored, _ = np.histogram(class1_all, bins=bins)

    valid_bins = hist_total >= min_pixels
    proportions = np.zeros_like(hist_total, dtype=float)
    proportions[valid_bins] = hist_discolored[valid_bins] / hist_total[valid_bins]

    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    fig, ax = plt.subplots(figsize=(9, 6))

    # --- QUARTILE BACKGROUND SHADING ---
    # Computed here so the shading sits behind the bars (zorder=0)
    qr = None
    if show_quartiles and np.sum(valid_bins) >= 4:
        qr = compute_quartile_stats(
            bin_centers=bin_centers[valid_bins],
            proportions=proportions[valid_bins],
            totals=hist_total[valid_bins].astype(float)
        )

    if qr is not None:
        q25_awc = np.percentile(bin_centers[valid_bins], 25)
        q75_awc = np.percentile(bin_centers[valid_bins], 75)
        x_min, x_max = bins.min(), bins.max()

        # Q1 band — low AWC (light grey)
        ax.axvspan(x_min, q25_awc,
                   color="#cccccc", alpha=0.35, zorder=0, label="_nolegend_")
        # IQR band — mid AWC (no shading)
        ax.axvspan(q25_awc, q75_awc,
                   color="#bbbbbb", alpha=0.0, zorder=0, label="_nolegend_")
        # Q3 band — high AWC (light grey)
        ax.axvspan(q75_awc, x_max,
                   color="#cccccc", alpha=0.35, zorder=0, label="_nolegend_")

        # Quartile boundary lines
        for xv, lbl in [(q25_awc, "Q1/Q2"), (q75_awc, "Q2/Q3")]:
            ax.axvline(xv, color="#333333", linewidth=0.9,
                       linestyle=":", zorder=1, label="_nolegend_")
            ax.text(xv, 0, f" {lbl}", fontsize=7, color="#333333",
                    va="bottom", ha="left", rotation=90,
                    transform=ax.get_xaxis_transform())

    bars = ax.bar(
        bin_centers[valid_bins],
        proportions[valid_bins],
        width=bin_width * 0.9,
        color="#ee9b00",  # "#ca6702",
        edgecolor="black",
        alpha=0.85,
        label="Proportion discolored (bin)"
    )

    if vertical_fraction_labels:
        # --- VERTICAL FRACTION LABELS: DISCOLOR / TOTAL ---
        active_discolored = hist_discolored[valid_bins]
        active_total = hist_total[valid_bins]

        labels = [f"{int(disc):,} / {int(tot):,}" for disc, tot in zip(active_discolored, active_total)]

        ax.bar_label(
            bars,
            labels=labels,
            label_type='edge',
            padding=5,
            rotation=90,
            fontsize=8.0,
            color="#111111",
            weight='semibold'
        )

    # --- WEIGHTED LOGIT REGRESSION OVERLAY ---
    reg_result = None
    if show_regression and np.sum(valid_bins) >= 3:
        reg_result = fit_weighted_logit_slope(
            bin_centers=bin_centers[valid_bins],
            proportions=proportions[valid_bins],
            totals=hist_total[valid_bins].astype(float)
        )

    if reg_result is not None:
        ax.plot(
            reg_result["x_fit"],
            reg_result["fitted_proportions"],
            color="black",
            linewidth=2.2,
            zorder=5,
            label="_nolegend_"   # handled manually below
        )

        # Shaded 95% CI band around the fitted line (propagated from slope CI)
        logit_lo = reg_result["slope_ci_lo"] * reg_result["x_fit"] + reg_result["intercept"]
        logit_hi = reg_result["slope_ci_hi"] * reg_result["x_fit"] + reg_result["intercept"]
        p_lo = 1.0 / (1.0 + np.exp(-logit_lo))
        p_hi = 1.0 / (1.0 + np.exp(-logit_hi))

        ax.fill_between(
            reg_result["x_fit"], p_lo, p_hi,
            color="black", alpha=0.10, zorder=4
        )

        # Format p-value string
        pv = reg_result["p_value"]
        if pv < 0.001:
            p_str = "p < 0.001"
        elif pv < 0.01:
            p_str = f"p = {pv:.3f}"
        else:
            p_str = f"p = {pv:.2f}"

        # Annotation text box (top-right corner)
        stats_text = (
            f"Weighted logit regression\n"
            f"β = {reg_result['slope']:.4f} logit mm⁻¹\n"
            f"OR per 50 mm AWC = {reg_result['or_per_50']:.3f}\n"
            f"{p_str}  (n = {int(np.sum(valid_bins))} bins)"
        )
        ax.text(
            0.97, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', alpha=0.85),
            color="black"
        )

        # Manual legend entries
        bar_patch = plt.Rectangle((0, 0), 1, 1, fc="#8B4513", ec="black", alpha=0.85)
        reg_line = mlines.Line2D([], [], color="black", linewidth=2.2, label="Weighted logit fit (95% CI)")
        ax.legend(
            handles=[bar_patch, reg_line],
            labels=["Proportion discolored (bin)", "Weighted logit fit (95% CI)"],
            loc="upper right",
            fontsize=8.5,
            framealpha=0.9,
            bbox_to_anchor=(0.97, 0.72)   # sits just below the stats box
        )

    # --- QUARTILE ANNOTATION BOX ---
    if qr is not None:
        ci_str = (f"[{qr['rr_ci_lo']:.2f}–{qr['rr_ci_hi']:.2f}]"
                  if not np.isnan(qr['rr_ci_lo']) else "[CI n/a]")

        q_text = (
            f"Quartile risk summary\n"
            f"Q1 AWC {qr['q1_awc_range'][0]:.0f}–{qr['q1_awc_range'][1]:.0f} mm"
            f"  p = {qr['q1_prop']:.3f}  (n={qr['q1_n']:,})\n"
            f"IQR AWC {qr['q2_awc_range'][0]:.0f}–{qr['q2_awc_range'][1]:.0f} mm"
            f"  p = {qr['q2_prop']:.3f}  (n={qr['q2_n']:,})\n"
            f"Q3 AWC {qr['q3_awc_range'][0]:.0f}–{qr['q3_awc_range'][1]:.0f} mm"
            f"  p = {qr['q3_prop']:.3f}  (n={qr['q3_n']:,})\n"
            f"RR Q1 vs Q3 = {qr['rr_q1_vs_q3']:.2f} {ci_str}\n"
            f"RR Q1 vs IQR = {qr['rr_q1_vs_q2']:.2f}  |  "
            f"ARD = {qr['arr_q1_vs_q3']:+.3f}"
        )

        # Position below regression box if both shown, otherwise top-right
        y_anchor = 0.685 if reg_result is not None else 0.97
        ax.text(
            0.97, y_anchor, q_text,
            transform=ax.transAxes,
            fontsize=8.0,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#333333', alpha=0.85),
            color="#333333"
        )

    # ax.set_title(f"Proportion of Discoloration by AWC\n{dataset} · {depth} · {disco_date}", fontsize=14, pad=15)
    # ax.set_xlabel("AWC (mm)", fontsize=24)
    # ax.set_ylabel("Proportion Discolored", fontsize=24)

    ax.tick_params(axis='both', which='major', labelsize=30)

    # ax.set_xlim(bins.min(), bins.max())
    # ax.set_ylim(0, ylim * 1.22 if ylim < 0.8 else 1.18)
    ax.set_ylim(0, 0.85)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(pad=0.015)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.015)
    else:
        plt.show()

    plt.close(fig)


def build_ptf_boxplot_grid(
    image_dir,
    output_filename=None,
    group_order=('1m', '2m', 'mrd'),
    column_headers=None,
    dpi=300,
    cell_width=3.0,
    ncols=5,
    block_spacer_frac=0.15,
):
    """
    Combine per-PTF boxplot images (named proportion_<ptf>_<group>_<date>.png)
    into a single grid figure, with groups (1m/2m/mrd) as rows and PTFs as
    columns. PTFs are split across `ncols` columns per row-set; if there
    are more PTFs than fit in one row-set, additional sets are stacked
    vertically beneath the first (e.g. 10 PTFs with ncols=5 becomes two
    stacked 3-row x 5-col blocks).

    Each plot is labeled with its group (1 m / 2 m / MRD) directly in its
    top-right corner instead of via a shared row label. Each block gets
    its own "Proportion Discolored" y-label, vertically centered on that
    block's row range.

    Parameters
    ----------
    image_dir : str
        Directory containing the source PNG images. The combined grid is
        also saved here by default.
    output_filename : str, optional
        Full path for the output PNG. Defaults to
        '<image_dir>/combined_grid_comparison.png'.
    group_order : tuple of str, optional
        Order of the groups (rows within each block). Defaults to
        ('1m', '2m', 'mrd').
    column_headers : dict, optional
        Mapping from group key to display name for the in-plot group
        label. Defaults to {'1m': '1 m', '2m': '2 m', 'mrd': 'MRD'}.
    dpi : int, optional
        Resolution of the saved figure. Defaults to 300.
    cell_width : float, optional
        Width in inches of each grid column. Row height is derived from
        this using the source images' native aspect ratio. Defaults to 3.0.
    ncols : int, optional
        Number of PTF columns shown per row-block. Defaults to 5.
    block_spacer_frac : float, optional
        Height of the gap row inserted between blocks, expressed as a
        fraction of a normal row's height. Defaults to 0.15.

    Returns
    -------
    str or None
        The path to the saved combined image, or None if no matching
        images were found.
    """
    if column_headers is None:
        column_headers = {'1m': '1 m', '2m': '2 m', 'mrd': 'MRD'}
    if output_filename is None:
        output_filename = os.path.join(image_dir, 'combined_grid_comparison.png')
    group_pattern = '|'.join(re.escape(g) for g in group_order)
    pattern = re.compile(rf"proportion_(.*)_({group_pattern})_(\d{{4}}-\d{{2}}-\d{{2}})\.png")

    parsed_files = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            match = pattern.match(filename)
            if match:
                ptf = match.group(1)
                group = match.group(2)
                parsed_files.append({
                    'filename': filename,
                    'filepath': os.path.join(image_dir, filename),
                    'ptf': ptf,
                    'group': group
                })
    ptfs = sorted(list(set(f['ptf'] for f in parsed_files)), key=str.lower)
    if not ptfs:
        print("No images matching the pattern were found. Please check your folder and filenames.")
        return None

    n_ptfs = len(ptfs)
    n_groups = len(group_order)
    n_blocks = math.ceil(n_ptfs / ncols)

    # Build row layout: n_groups real rows per block, with a tall spacer
    # row between blocks (not before the first or after the last)
    row_types = []
    height_ratios = []
    block_row_start = []
    for block in range(n_blocks):
        block_row_start.append(len(row_types))
        for _ in range(n_groups):
            row_types.append('plot')
            height_ratios.append(1.0)
        if block < n_blocks - 1:
            row_types.append('spacer')
            height_ratios.append(block_spacer_frac)
    nrows = len(row_types)

    sample_img = mpimg.imread(parsed_files[0]['filepath'])
    img_h, img_w = sample_img.shape[0], sample_img.shape[1]
    img_aspect = img_h / img_w
    cell_height = cell_width * img_aspect

    # Fixed-inch margins, converted to figure fractions below, so spacing
    # stays visually consistent regardless of overall grid shape
    title_pad_inches = 1.5     # room for PTF column titles above row 0
    xlabel_pad_inches = 0.45   # room for shared supxlabel below last block
    ylabel_pad_inches = 0.45   # room for per-block "Proportion Discolored" labels

    plot_height_inches = cell_height * sum(height_ratios)
    plot_width_inches = cell_width * ncols

    fig_height = plot_height_inches + title_pad_inches + xlabel_pad_inches
    fig_width = plot_width_inches + ylabel_pad_inches

    with plt.style.context('default'):
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(fig_width, fig_height),
            facecolor='white',
            gridspec_kw={'height_ratios': height_ratios}
        )

        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]

        # Turn off all spacer-row axes entirely (pure whitespace)
        for row_idx, rtype in enumerate(row_types):
            if rtype == 'spacer':
                for col_idx in range(ncols):
                    axes[row_idx][col_idx].axis('off')

        for block in range(n_blocks):
            start_row = block_row_start[block]
            ptf_chunk = ptfs[block * ncols: (block + 1) * ncols]
            for local_row, group in enumerate(group_order):
                actual_row = start_row + local_row
                for col_idx in range(ncols):
                    ax = axes[actual_row][col_idx]
                    ax.set_facecolor('white')

                    if col_idx >= len(ptf_chunk):
                        ax.axis('off')
                        continue

                    ptf = ptf_chunk[col_idx]
                    matching_file = [f for f in parsed_files if f['ptf'] == ptf and f['group'] == group]
                    if matching_file:
                        img_path = matching_file[0]['filepath']
                        img = mpimg.imread(img_path)
                        ax.imshow(img)
                    ax.axis('off')

                    if local_row == 0:
                        display_ptf = ptf.replace('_', ' ').title().replace(' And ', ' and ')
                        ax.set_title(display_ptf, fontsize=14, pad=4)

                    # Group label (1 m / 2 m / MRD) in the top-right corner
                    # of each plot, instead of a shared row-side label
                    display_group = column_headers.get(group, group)
                    ax.text(0.96, 0.94, display_group, transform=ax.transAxes,
                            fontsize=11, va='top', ha='right', fontweight='semibold',
                            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                                      edgecolor='none', alpha=0.75))

        # Convert fixed-inch margins to figure-fraction coordinates
        top_fraction = 1 - (title_pad_inches / fig_height)
        bottom_fraction = xlabel_pad_inches / fig_height
        left_fraction = ylabel_pad_inches / fig_width

        xlabel_y = (xlabel_pad_inches * 0.35) / fig_height
        ylabel_x = (ylabel_pad_inches * 0.25) / fig_width

        # Shared x-label since PTF columns aren't split across blocks
        fig.supxlabel("AWC (mm)", fontsize=16, y=xlabel_y)

        plt.subplots_adjust(hspace=0.025, wspace=0.05,
                             left=left_fraction, bottom=bottom_fraction, top=top_fraction)

        # One "Proportion Discolored" y-label per block, vertically
        # centered on that block's own row range. Must be placed after
        # subplots_adjust so axes positions (get_position) reflect the
        # final layout.
        for block in range(n_blocks):
            start_row = block_row_start[block]
            end_row = start_row + n_groups - 1
            top_pos = axes[start_row][0].get_position().y1
            bottom_pos = axes[end_row][0].get_position().y0
            center_y = (top_pos + bottom_pos) / 2
            fig.text(ylabel_x, center_y, "Proportion Discolored", fontsize=16,
                      ha='center', va='center', rotation=90)

        plt.savefig(output_filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close(fig)

    return output_filename


def build_ptf_boxplot_grid_horizontal(
    image_dir,
    output_filename=None,
    group_order=('1m', '2m', 'mrd'),
    column_headers=None,
    dpi=300,
    cell_width=3.0,
    n_blocks=2,
    block_spacer_frac=0.15,
):
    """
    Combine per-PTF boxplot images (named proportion_<ptf>_<group>_<date>.png)
    into a single grid figure, with PTFs as rows and groups as columns.

    To keep tall grids manageable, PTFs can be split into side-by-side
    column-blocks (each block repeating the full set of group columns),
    e.g. 10 PTFs with n_blocks=2 becomes a 5-row x 6-col grid instead of
    a 10-row x 3-col grid. Each block gets its own "AWC (mm)" x-label
    centered beneath it.

    Parameters
    ----------
    image_dir : str
        Directory containing the source PNG images. The combined grid is
        also saved here by default.
    output_filename : str, optional
        Full path for the output PNG. Defaults to
        '<image_dir>/combined_grid_comparison.png'.
    group_order : tuple of str, optional
        Order of the groups (repeated per block). Defaults to ('1m', '2m', 'mrd').
    column_headers : dict, optional
        Mapping from group key to display name for column titles.
        Defaults to {'1m': '1 m', '2m': '2 m', 'mrd': 'MRD'}.
    dpi : int, optional
        Resolution of the saved figure. Defaults to 300.
    cell_width : float, optional
        Width in inches of each grid column. Row height is derived from
        this using the source images' native aspect ratio. Defaults to 3.0.
    n_blocks : int, optional
        Number of side-by-side column-blocks to split the PTFs across.
        Defaults to 2.
    block_spacer_frac : float, optional
        Width of the gap column inserted between blocks, expressed as a
        fraction of a normal column's width. Defaults to 0.25.

    Returns
    -------
    str or None
        The path to the saved combined image, or None if no matching
        images were found.
    """
    if column_headers is None:
        column_headers = {'1m': '1 m', '2m': '2 m', 'mrd': 'MRD'}
    if output_filename is None:
        output_filename = os.path.join(image_dir, 'combined_grid_comparison.png')
    group_pattern = '|'.join(re.escape(g) for g in group_order)
    pattern = re.compile(rf"proportion_(.*)_({group_pattern})_(\d{{4}}-\d{{2}}-\d{{2}})\.png")

    parsed_files = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            match = pattern.match(filename)
            if match:
                ptf = match.group(1)
                group = match.group(2)
                parsed_files.append({
                    'filename': filename,
                    'filepath': os.path.join(image_dir, filename),
                    'ptf': ptf,
                    'group': group
                })
    ptfs = sorted(list(set(f['ptf'] for f in parsed_files)), key=str.lower)
    if not ptfs:
        print("No images matching the pattern were found. Please check your folder and filenames.")
        return None

    n_ptfs = len(ptfs)
    n_groups = len(group_order)
    rows_per_block = math.ceil(n_ptfs / n_blocks)
    nrows = rows_per_block

    # Build column layout: n_groups real columns per block, with a narrow
    # spacer column between blocks (not before the first or after the last)
    col_types = []
    width_ratios = []
    block_col_start = []
    for block in range(n_blocks):
        block_col_start.append(len(col_types))
        for _ in range(n_groups):
            col_types.append('plot')
            width_ratios.append(1.0)
        if block < n_blocks - 1:
            col_types.append('spacer')
            width_ratios.append(block_spacer_frac)
    ncols = len(col_types)

    sample_img = mpimg.imread(parsed_files[0]['filepath'])
    img_h, img_w = sample_img.shape[0], sample_img.shape[1]
    img_aspect = img_h / img_w
    cell_height = cell_width * img_aspect

    # Fixed-inch margins, converted to figure fractions below, so spacing
    # stays visually consistent regardless of overall grid shape
    title_pad_inches = 1
    xlabel_pad_inches = 0.05
    ylabel_pad_inches = 0.9

    plot_height_inches = cell_height * nrows
    plot_width_inches = cell_width * sum(width_ratios)

    fig_height = plot_height_inches + title_pad_inches + xlabel_pad_inches
    fig_width = plot_width_inches + ylabel_pad_inches

    with plt.style.context('default'):
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(fig_width, fig_height),
            facecolor='white',
            gridspec_kw={'width_ratios': width_ratios}
        )

        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]

        for row_idx in range(nrows):
            for actual_col, ctype in enumerate(col_types):
                if ctype == 'spacer':
                    axes[row_idx][actual_col].axis('off')

        for block in range(n_blocks):
            start_col = block_col_start[block]
            for local_row in range(rows_per_block):
                ptf_idx = block * rows_per_block + local_row
                for col_idx, group in enumerate(group_order):
                    actual_col = start_col + col_idx
                    ax = axes[local_row][actual_col]
                    ax.set_facecolor('white')

                    if ptf_idx >= n_ptfs:
                        ax.axis('off')
                        continue

                    ptf = ptfs[ptf_idx]
                    matching_file = [f for f in parsed_files if f['ptf'] == ptf and f['group'] == group]
                    if matching_file:
                        img_path = matching_file[0]['filepath']
                        img = mpimg.imread(img_path)
                        ax.imshow(img)
                    ax.axis('off')

                    if local_row == 0:
                        ax.set_title(column_headers.get(group, group), fontsize=14, pad=4)
                    if col_idx == 0:
                        display_name = ptf.replace('_', ' ').title().replace(' And ', ' and ')
                        ax.text(-0.05, 0.5, display_name, transform=ax.transAxes,
                                fontsize=12, va='center', ha='right', rotation=90)

        # Convert fixed-inch margins to figure-fraction coordinates
        top_fraction = 1 - (title_pad_inches / fig_height)
        bottom_fraction = xlabel_pad_inches / fig_height
        left_fraction = ylabel_pad_inches / fig_width

        xlabel_y = (xlabel_pad_inches * 0.35) / fig_height
        ylabel_x = (ylabel_pad_inches * 0.25) / fig_width

        # Shared y-label still spans the whole figure since all rows align
        fig.supylabel("Proportion Discolored", fontsize=16, x=ylabel_x)

        plt.subplots_adjust(hspace=0.01, wspace=0.05,
                             left=left_fraction, bottom=bottom_fraction, top=top_fraction)

        # One "AWC (mm)" x-label per block, centered under that block's
        # own columns. Must be placed after subplots_adjust so axes
        # positions (get_position) reflect the final layout.
        for block in range(n_blocks):
            start_col = block_col_start[block]
            end_col = start_col + n_groups - 1
            left_pos = axes[0][start_col].get_position().x0
            right_pos = axes[0][end_col].get_position().x1
            center_x = (left_pos + right_pos) / 2
            fig.text(center_x, xlabel_y, "AWC (mm)", fontsize=16, ha='center', va='top')

        plt.savefig(output_filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close(fig)

    return output_filename


def build_ptf_boxplot_grid_vertical(
    image_dir,
    output_filename=None,
    group_order=('1m', '2m', 'mrd'),
    column_headers=None,
    dpi=300,
    cell_width=3.0,
):
    """
    Combine per-PTF boxplot images (named proportion_<ptf>_<group>_<date>.png)
    into a single grid figure, with PTFs as rows and groups as columns.
    ...
    """
    if column_headers is None:
        column_headers = {'1m': '1 m', '2m': '2 m', 'mrd': 'MRD'}
    if output_filename is None:
        output_filename = os.path.join(image_dir, 'combined_grid_comparison.png')
    group_pattern = '|'.join(re.escape(g) for g in group_order)
    pattern = re.compile(rf"proportion_(.*)_({group_pattern})_(\d{{4}}-\d{{2}}-\d{{2}})\.png")

    parsed_files = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            match = pattern.match(filename)
            if match:
                ptf = match.group(1)
                group = match.group(2)
                parsed_files.append({
                    'filename': filename,
                    'filepath': os.path.join(image_dir, filename),
                    'ptf': ptf,
                    'group': group
                })
    ptfs = sorted(list(set(f['ptf'] for f in parsed_files)), key=str.lower)
    if not ptfs:
        print("No images matching the pattern were found. Please check your folder and filenames.")
        return None

    ncols = len(group_order)
    nrows = len(ptfs)

    sample_img = mpimg.imread(parsed_files[0]['filepath'])
    img_h, img_w = sample_img.shape[0], sample_img.shape[1]
    img_aspect = img_h / img_w
    cell_height = cell_width * img_aspect

    # Reserve extra height for column titles so they aren't clipped when
    # the figure is tightly sized to match the image aspect ratio
    title_pad_inches = 0.35
    fig_height = cell_height * nrows + title_pad_inches

    with plt.style.context('default'):
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(cell_width * ncols, fig_height),
            facecolor='white'
        )

        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]

        for row_idx, ptf in enumerate(ptfs):
            for col_idx, group in enumerate(group_order):
                ax = axes[row_idx][col_idx]
                ax.set_facecolor('white')
                matching_file = [f for f in parsed_files if f['ptf'] == ptf and f['group'] == group]
                if matching_file:
                    img_path = matching_file[0]['filepath']
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                ax.axis('off')
                if row_idx == 0:
                    ax.set_title(column_headers.get(group, group), fontsize=14, pad=4)
                if col_idx == 0:
                    display_name = ptf.replace('_', ' ').title().replace(' And ', ' and ')
                    ax.text(-0.05, 0.5, display_name, transform=ax.transAxes,
                            fontsize=12, va='center', ha='right', rotation=90)

        fig.supxlabel("AWC (mm)", fontsize=16, y=0.04)
        fig.supylabel("Proportion Discolored", fontsize=16, x=0.04)

        top_fraction = 1 - (title_pad_inches / fig_height)
        plt.subplots_adjust(hspace=0.01, wspace=0.05, left=0.12, bottom=0.05, top=top_fraction)
        plt.savefig(output_filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close(fig)

    return output_filename


if __name__ == "__main__":
    pass