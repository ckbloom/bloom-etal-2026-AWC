"""
utils.py
========
Shared I/O, data-loading, and raster helpers for the AWC pipeline:
raster clipping/sorting, soil-cube loading, the van Genuchten water
retention model, and AWC integration/export.
"""

import os
import glob
import shutil
import logging
from pathlib import Path
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import numpy as np
from shapely.geometry import mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def clip_tif(tiff_path, shapefile_path, output_dir):
    """
    Clip a GeoTIFF raster to a shapefile's geometry and save the result.

    Reprojects the shapefile to the raster's CRS if they differ. Output
    filename is derived as "<tiff_stem>_<shapefile_stem><tiff_suffix>".

    Parameters:
        tiff_path      : path to the input GeoTIFF
        shapefile_path : path to the clipping shapefile
        output_dir     : directory to write the clipped raster into (created if missing)
    """
    # Convert string paths to Path objects for easy manipulation
    tiff_path = Path(tiff_path)
    shapefile_path = Path(shapefile_path)
    output_dir = Path(output_dir)

    # Create the output directory if it doesn't exist yet
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dynamically build the new filename: e.g., "tif1.tif" + "basel.shp" -> "tif1_basel.tif"
    new_filename = f"{tiff_path.stem}_{shapefile_path.stem}{tiff_path.suffix}"
    final_output_path = output_dir / new_filename

    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Load the TIFF file
    raster = rxr.open_rasterio(tiff_path, masked=True)

    # Match CRS (Coordinate Reference System)
    if raster.rio.crs != gdf.crs:
        print(
            f"Projecting shapefile from {gdf.crs} to match raster CRS: {raster.rio.crs}"
        )
        gdf = gdf.to_crs(raster.rio.crs)

    # Clip the raster using the shapefile's geometry
    clipped_raster = raster.rio.clip(gdf.geometry.apply(mapping), gdf.crs)

    # Save the clipped raster to the dynamically generated path
    clipped_raster.rio.to_raster(final_output_path)

    print(f"Successfully clipped and saved to: {final_output_path}")


def sort_images_by_name(source_dir, target_dir, keyword):
    """
    Move .png and .csv files whose filename contains `keyword` from
    source_dir to target_dir.

    Parameters:
        source_dir : directory to scan (non-recursive)
        target_dir : directory to move matching files into (created if missing)
        keyword    : case-insensitive substring to match against filenames
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    target_path.mkdir(parents=True, exist_ok=True)

    moved_count = 0

    for file_path in source_path.glob('*'):
        if file_path.suffix.lower() in ['.png', '.csv']:
            if keyword.lower() in file_path.name.lower():
                destination = target_path / file_path.name

                shutil.move(str(file_path), str(destination))

                print(f'Moved {file_path.name} to  {target_path.name}')
                moved_count += 1
    print(f'Completed. Total files moved = {moved_count}')


def calculate_awc_mrd(layer_awc_mm, mrd_da, ds_depths):
    """
    Integrate per-layer AWC down to the maximum rooting depth (MRD),
    pro-rating the AWC of the layer the rooting depth falls within.

    Parameters:
        layer_awc_mm : per-layer AWC (mm), with a `thick_mm` coordinate
        mrd_da       : maximum rooting depth (cm)
        ds_depths    : upper depth of each layer (cm)

    Returns:
        AWC integrated to MRD (mm), NaN where mrd_da is NaN
    """
    thick_mm = layer_awc_mm.thick_mm
    thick_cm = thick_mm / 10
    lower_cm = ds_depths + thick_cm

    # Full layers
    full_layer = layer_awc_mm.where(mrd_da >= lower_cm, 0)

    # Partial layers
    intersect_mask = (mrd_da > ds_depths) & (mrd_da < lower_cm)
    awc_per_mm = layer_awc_mm / thick_mm.where(thick_mm > 0)
    partial_thick_mm = (mrd_da - ds_depths) * 10
    partial_layer = (awc_per_mm * partial_thick_mm).where(intersect_mask, 0)

    combined = full_layer + partial_layer

    # Re-apply the mrd NaN mask: pixels with no rooting depth stay NaN
    combined = combined.where(mrd_da.notnull())

    return combined.sum(dim='depth', min_count=1)


def load_soil_data_cube(source_dir, depth_ranges, soil_names_map, load_mrd=True,
                        chunk_size=512):
    """
    Load per-depth soil raster layers into a single xarray Dataset cube.

    Parameters:
        source_dir     : directory containing the source GeoTIFFs
        depth_ranges   : list of "<upper>_<lower>" depth range strings (cm),
                         e.g. ["0_5", "5_15", ...]
        soil_names_map : mapping of standard variable name -> filename search key
                         (e.g. {"sand": "SAND", "mrd": "MRD"})
        load_mrd       : if True, also load and return the MRD raster
        chunk_size     : dask chunk size (pixels) for the x/y dimensions

    Returns:
        full_cube         : xarray.Dataset with a `depth` dimension and a
                            `thick_mm` depth coordinate
        full_cube, mrd_da : as above, plus the MRD DataArray reprojected to
                            match full_cube, if load_mrd is True
    """
    depth_ds_list = []
    thicknesses = []
    for dr in depth_ranges:
        upper, lower = map(int, dr.split("_"))
        thicknesses.append((lower - upper) * 10)
        var_da_list = []
        for std_name, search_key in soil_names_map.items():
            pattern = os.path.join(source_dir, f"*{search_key}*{dr}*.tif")
            files = glob.glob(pattern)
            if not files:
                continue
            da = rxr.open_rasterio(
                files[0], masked=True, chunks={"x": chunk_size, "y": chunk_size}
            ).astype(np.float32)
            da = da.squeeze().drop_vars("band", errors="ignore")
            da.name = std_name
            var_da_list.append(da)
        if var_da_list:
            ds_depth = xr.merge(var_da_list, compat="override")
            ds_depth = ds_depth.assign_coords(depth=upper)
            depth_ds_list.append(ds_depth)
    full_cube = xr.concat(depth_ds_list, dim="depth", coords="minimal", compat="override")
    full_cube = full_cube.assign_coords(
        thick_mm=("depth", np.array(thicknesses, dtype=np.float32))
    )
    if load_mrd:
        # Open MRD data
        mrd_path = os.path.join(source_dir, f"{soil_names_map['mrd']}.tif")
        mrd_da = rxr.open_rasterio(mrd_path, masked=True).squeeze()
        mrd_da = mrd_da.rio.reproject_match(full_cube["sand"])
        mrd_da = mrd_da.where(mrd_da > 0)

        return full_cube, mrd_da
    else:
        return full_cube


def mvg_theta(h_cm, ths, thr, alpha_cm, n):
    """
    Van Genuchten volumetric water content at a given suction head.
    θ(ψ) = θr + (θs − θr) / (1 + (α·ψ)^n)^m     m = 1 − 1/n

    Parameters:
        h_cm     : positive suction head (cm)
        ths      : saturated water content (θs)
        thr      : residual water content (θr)
        alpha_cm : van Genuchten α (cm⁻¹)
        n        : shape parameter, must be > 1

    Returns:
        θ(ψ) — volumetric water content (cm³ cm⁻³)
    """
    n_safe = (xr.where(n <= 1.0, np.nan, n) if isinstance(n, xr.DataArray)
              else np.where(n <= 1.0, np.nan, n))  # n must be > 1; invalid values propagate as NaN
    m = 1.0 - 1.0 / n_safe
    denom = (1.0 + (alpha_cm * h_cm) ** n_safe) ** m
    return (thr + (ths - thr) / denom).astype(np.float32)


def export_awc(ds, mrd, ptf_name, nodata_value=-9999, output_dir=None):
    """
    Integrate AWC over depth and export 1 m / 2 m (/ MRD) totals as GeoTIFFs.

    Applies a gravel-fraction correction to layer AWC, sums over depth for
    the 1 m and 2 m totals, and (if mrd is given) integrates to the maximum
    rooting depth via calculate_awc_mrd. Rasters are reprojected to
    EPSG:2056, encoded as int16 with nodata_value, and written to output_dir.

    Parameters:
        ds           : xarray.Dataset with awc_vol, gravel, thick_mm, depth
        mrd          : maximum rooting depth DataArray, or None to skip
        ptf_name     : PTF name, used in output filenames and log messages
        nodata_value : integer nodata/fill value for the exported rasters
        output_dir   : directory to write the GeoTIFFs into
    """

    # ------------------------------------------------------------------
    # Layer AWC in mm  (gravel correction)
    # ------------------------------------------------------------------
    gravel_frac = ds["gravel"] / 100.0
    layer_awc_mm = ds["awc_vol"] * (1 - gravel_frac) * ds.thick_mm
    layer_awc_mm.coords["thick_mm"] = ds.thick_mm

    # ------------------------------------------------------------------
    # Depth-integrated totals
    # ------------------------------------------------------------------
    logging.info(f"Integrating AWC {ptf_name} over depth")
    awc_2m = layer_awc_mm.sum(dim="depth", min_count=1)
    awc_1m = layer_awc_mm.where(ds.depth < 100).sum(dim="depth", min_count=1)

    if mrd is not None:
        awc_mrd = calculate_awc_mrd(layer_awc_mm, mrd, ds.depth)

        outputs = {
            f"awc_2m_{ptf_name}.tif": awc_2m,
            f"awc_1m_{ptf_name}.tif": awc_1m,
            f"awc_mrd_{ptf_name}.tif": awc_mrd,
        }

    else:
        outputs = {
            f"awc_2m_{ptf_name}.tif": awc_2m,
            f"awc_1m_{ptf_name}.tif": awc_1m,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    logging.info(f"Exporting results of {ptf_name}")
    with np.errstate(divide="ignore", invalid="ignore"):
        for filename, da in outputs.items():
            da_final = da.rio.reproject("EPSG:2056")
            da_encoded = da_final.fillna(nodata_value).astype("int16")
            da_encoded.rio.write_nodata(nodata_value, inplace=True)
            da_encoded.attrs["scale_factor"] = 1.0
            da_encoded.attrs["add_offset"] = 0.0
            da_encoded.attrs["_FillValue"] = nodata_value
            da_encoded.rio.to_raster(
                os.path.join(output_dir, filename),
                tiled=True, windowed=True, compress="lzw", dtype="int16",
            )
            logging.info(f"Exported {filename}")


def scale_and_save_as_int(input_path, output_path, dtype="int16"):
    """
    Reads a TIF with rioxarray, scales it by the largest power-of-10 factor
    (100, 10, or 1) that preserves up to 2 decimal points without
    overflowing the target int dtype, then writes it back out as that dtype.

    A new nodata value (the dtype's min value) is set for the output,
    since the original nodata isn't representable in an int type.
    The applied scale_factor is stored in the output's metadata so values
    can be recovered later (true_value = stored_value / scale_factor).
    """
    da = rxr.open_rasterio(input_path, masked=True)  # nodata -> NaN

    dtype_info = np.iinfo(dtype)
    int_min, int_max = dtype_info.min, dtype_info.max

    valid = da.values[~np.isnan(da.values)]
    if valid.size == 0:
        raise ValueError("No valid data found in raster")

    max_abs = np.nanmax(np.abs(valid))

    # Pick the largest scale factor that keeps the scaled max within range,
    # leaving int_min reserved as the nodata sentinel.
    scale_factor = 1
    for candidate in (10000, 1000, 100, 10, 1):
        if max_abs * candidate <= int_max:
            scale_factor = candidate
            break

    nodata_value = int_min

    scaled = (da * scale_factor).round()
    scaled = scaled.fillna(nodata_value).astype(dtype)

    scaled.rio.write_nodata(nodata_value, inplace=True)
    scaled.attrs["scale_factor"] = 1.0 / scale_factor  # GDAL convention: true = stored * scale_factor
    scaled.rio.to_raster(output_path)

    return output_path, scale_factor, nodata_value


def scale_directory(input_dir, output_dir, dtype="int16"):
    """Runs scale_and_save_as_int() over every .tif in input_dir, writing
    each result to the same-named file under output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    tif_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".tif", ".tiff"))]

    results = []
    for fname in tif_files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        results.append(scale_and_save_as_int(in_path, out_path, dtype=dtype))

    return results


if __name__ == "__main__":
    pass
