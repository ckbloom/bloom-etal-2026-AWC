import os
import numpy as np
import rasterio

PTF_COLORS = [
    "#378ADD",  # blue
    "#1D9E75",  # teal
    "#D85A30",  # coral
    "#D4537E",  # pink
    "#888780",  # gray
    "#7F77DD",  # purple
    "#639922",  # green
    "#BA7517",  # amber
    "#E24B4A",  # red
]


def read_tif_as_array(filepath):
    """Read a GeoTIFF and return a flat 1-D array of valid (non-nodata) values."""
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    return data[np.isfinite(data)].ravel()


def make_boxplot(data_dir, depth, ptfs, ax, sample_limit=500_000):
    """Draw a box-and-whisker plot for all PTFs at a given depth on ax."""

    DEPTH_LABELS = {"1m": "1 m", "2m": "2 m", "mrd": "Maximum rooting depth"}

    all_data = []
    labels = []

    for ptf in ptfs:
        fname = f"awc_{depth}_{ptf}.tif"
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found, skipping.")
            continue

        arr = read_tif_as_array(fpath)

        # Sub-sample for speed if arrays are very large
        if len(arr) > sample_limit:
            rng = np.random.default_rng(42)
            arr = rng.choice(arr, size=sample_limit, replace=False)

        all_data.append(arr)

        # --- UPDATED: Split by underscore and take only the first word ---
        first_word = ptf.split("_")[0]
        labels.append(first_word)

    if not all_data:
        return

    # --- Calculate overall mean and standard deviation ---
    pooled_data = np.concatenate(all_data)
    overall_mean = np.mean(pooled_data)
    overall_std = np.std(pooled_data)

    # Plot the shaded band for overall +/- 1 SD (zorder=0 puts it behind boxes)
    ax.axhspan(overall_mean - overall_std, overall_mean + overall_std,
               color='gray', alpha=0.15, zorder=0)

    # Plot the horizontal line for overall mean
    ax.axhline(overall_mean, color='gray', linestyle='--', linewidth=1.5, zorder=1)

    # --- Boxplot setup ---
    bp = ax.boxplot(
        all_data,
        patch_artist=True,
        notch=False,
        whis=1.5,
        showfliers=False,  # hide individual outlier dots
        medianprops=dict(color="white", linewidth=2, zorder=4),
        zorder=2  # put boxes above the shaded band
    )

    for patch, color in zip(bp["boxes"], PTF_COLORS[: len(all_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    for element in ("whiskers", "caps"):
        for line, color in zip(
                bp[element],
                [c for c in PTF_COLORS[: len(all_data)] for _ in range(2)],
        ):
            line.set_color(color)
            line.set_linewidth(1.5)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=12, rotation=315, ha="left")
    ax.set_title(DEPTH_LABELS[depth], fontsize=15, fontweight="normal", pad=8)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("AWC (mm)", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


if __name__ == '__main__':
    pass

