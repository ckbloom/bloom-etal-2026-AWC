from pathlib import Path
from typing import Iterable, Union
import xarray as xr
import rioxarray
from rasterio.enums import Resampling


def combine_awc_by_ptf(
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        ptfs: Iterable[str],
        depths: Iterable[str] = ("1m", "2m"),
        pattern: str = "awc_{depth}_{ptf}.tif",
        resampling: Resampling = Resampling.bilinear,
) -> None:
    """Combines AWC rasters across a selected set of pedotransfer functions (PTFs)
    by computing the mean and standard deviation for each depth, and writes the
    results as new rasters.

    Args:
        input_dir: Directory containing the individual awc_<depth>_<ptf>.tif files.
        output_dir: Directory where the combined mean/sd rasters will be saved.
        ptfs: List of PTF names to include (e.g. ["wosten", "szabo", "toth", "teepe"]).
        depths: Depths to process. Defaults to ("1m", "2m").
        pattern: Filename pattern with {depth} and {ptf} placeholders, used to
          locate input files.
        resampling: Resampling method used when reprojecting mismatched grids
          onto the reference grid (default: bilinear). Use Resampling.nearest
          for categorical/discrete data.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    ptfs = list(ptfs)
    for depth in depths:
        files = []
        missing = []
        for ptf in ptfs:
            fpath = in_path / pattern.format(depth=depth, ptf=ptf)
            if fpath.exists():
                files.append(fpath)
            else:
                missing.append(fpath.name)
        if missing:
            print(f"[{depth}] Warning: missing files skipped: {missing}")
        if not files:
            print(f"[{depth}] No matching files found, skipping depth.")
            continue
        print(f"[{depth}] Combining {len(files)} PTFs: {[f.name for f in files]}")

        # Open each raster, masking nodata as NaN, and reproject-match to a
        # common reference grid if needed
        arrays = []
        ref = None
        for fpath in files:
            da = rioxarray.open_rasterio(fpath, masked=True).squeeze("band", drop=True)
            if ref is None:
                ref = da
            else:
                same_shape = da.shape == ref.shape
                same_transform = da.rio.transform() == ref.rio.transform()
                same_crs = da.rio.crs == ref.rio.crs
                if not (same_shape and same_transform and same_crs):
                    print(
                        f"[{depth}] Grid mismatch: reprojecting {fpath.name} "
                        f"to match {files[0].name}"
                    )
                    da = da.rio.reproject_match(ref, resampling=resampling)
                    # reproject_match can introduce tiny floating-point coordinate
                    # drift; force exact coordinate alignment with ref
                    da = da.assign_coords({"x": ref.x, "y": ref.y})
            arrays.append(da)

        stacked = xr.concat(arrays, dim="ptf")

        # Only keep pixels where every PTF has a valid value
        valid_mask = stacked.notnull().all(dim="ptf")
        stacked = stacked.where(valid_mask)

        mean_da = stacked.mean(dim="ptf", skipna=False)
        sd_da = stacked.std(dim="ptf", skipna=False)

        # Preserve CRS/spatial metadata
        mean_da = mean_da.rio.write_crs(ref.rio.crs)
        sd_da = sd_da.rio.write_crs(ref.rio.crs)

        out_nodata = -9999.0
        mean_da = mean_da.fillna(out_nodata).rio.write_nodata(out_nodata)
        sd_da = sd_da.fillna(out_nodata).rio.write_nodata(out_nodata)

        mean_path = out_path / f"awc_{depth}_mean.tif"
        sd_path = out_path / f"awc_{depth}_sd.tif"
        mean_da.astype("float32").rio.to_raster(mean_path)
        sd_da.astype("float32").rio.to_raster(sd_path)
        print(f"[{depth}] Saved {mean_path.name} and {sd_path.name}")


if __name__ == "__main__":
    pass
