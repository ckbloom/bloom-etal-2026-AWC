"""
Open-source equivalent of the ArcPy workflow, WITHOUT the h3 dependency.
Binarizes a 0-10000 scaled probability raster and computes the proportion
of "positive" (discoloration) cells per hexagon (manually generated flat-top
hexagon grid), for display across Switzerland.

Requires: rasterio, numpy, geopandas, shapely
    pip install rasterio numpy geopandas shapely

Speed strategy: rather than clipping the raster per-hexagon (what ArcGIS
zonal statistics does), we rasterize hexagon IDs onto the raster's own grid
ONCE, then use numpy.bincount to get per-hexagon counts and proportions in
a single vectorized pass.
"""

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
import geopandas as gpd
from shapely.geometry import Polygon


def _make_hexagon(cx, cy, r):
    """Flat-top hexagon vertices, circumradius r, centered at (cx, cy)."""
    angles_deg = np.arange(0, 360, 60)  # 0,60,120,180,240,300 -> flat-top orientation
    angles_rad = np.radians(angles_deg)
    xs = cx + r * np.cos(angles_rad)
    ys = cy + r * np.sin(angles_rad)
    return Polygon(zip(xs, ys))


def compute_hex_discoloration(
    in_raster_path,
    out_hex_path,
    boundary_path=None,
    threshold=4000,
    hex_area_km2=1,
    target_crs_epsg=2056,
):
    """
    Binarizes a 0-10000 scaled probability raster and computes the
    proportion of "positive" (discoloration) cells per hexagon, saving
    the result as a shapefile (or any format rasterio/geopandas infers
    from out_hex_path's extension).

    :param in_raster_path: path to the 0-10000 scaled probability raster.
    :param out_hex_path: output path for the hexagon layer (e.g. .shp).
    :param boundary_path: optional path to a polygon used to clip the hex
        grid. If None, the hex grid is generated over the raster's own
        extent instead.
    :param threshold: raster values strictly greater than this are
        treated as "positive" (discoloration) when binarizing.
    :param hex_area_km2: target area per hexagon in km^2, equivalent to
        ArcGIS GenerateTessellation's "Size" parameter.
    :param target_crs_epsg: EPSG code the hex grid is built directly in
        (and the CRS of the saved output). Defaults to 2056
        (CH1903+ / LV95).
    :return: the resulting GeoDataFrame (also written to out_hex_path).
    """
    # -------------------------------------------------------------
    # 1. READ + BINARIZE RASTER
    # -------------------------------------------------------------
    with rasterio.open(in_raster_path) as src:
        data = src.read(1)
        nodata = src.nodata
        transform = src.transform
        raster_crs = src.crs
        raster_bounds = src.bounds
        out_shape = (src.height, src.width)

    valid_mask = np.ones_like(data, dtype=bool)
    if nodata is not None:
        valid_mask &= (data != nodata)

    binary = np.where(data > threshold, 1, 0).astype(np.uint8)
    print(f"Binarized raster: {binary.sum():,} positive cells out of {valid_mask.sum():,} valid cells")

    # -------------------------------------------------------------
    # 2. GENERATE A FLAT-TOP HEXAGON GRID DIRECTLY IN THE TARGET CRS
    # -------------------------------------------------------------
    # For a regular flat-top hexagon with circumradius R (center-to-vertex):
    #   side length s = R
    #   area = (3 * sqrt(3) / 2) * R^2
    # Solve for R given a target area (units must match CRS units, i.e. meters for EPSG:2056)
    hex_area_m2 = hex_area_km2 * 1_000_000
    R = np.sqrt(hex_area_m2 / (1.5 * np.sqrt(3)))

    hex_width = 2 * R                  # horizontal extent, flat-top orientation
    hex_height = np.sqrt(3) * R        # vertical extent
    col_spacing = 1.5 * R              # horizontal distance between column centers
    row_spacing = hex_height           # vertical distance between row centers

    # Determine grid extent: use boundary if given (reprojected to target CRS), else raster extent
    if boundary_path:
        boundary_gdf = gpd.read_file(boundary_path).to_crs(target_crs_epsg)
        minx, miny, maxx, maxy = boundary_gdf.total_bounds
    else:
        minx, miny, maxx, maxy = transform_bounds(raster_crs, f"EPSG:{target_crs_epsg}", *raster_bounds)

    # Pad extent by one hex so edge hexagons fully cover the boundary
    minx -= hex_width
    miny -= hex_height
    maxx += hex_width
    maxy += hex_height

    n_cols = int(np.ceil((maxx - minx) / col_spacing)) + 1
    n_rows = int(np.ceil((maxy - miny) / row_spacing)) + 1

    polygons = []
    for col in range(n_cols):
        cx = minx + col * col_spacing
        row_offset = (row_spacing / 2) if (col % 2 == 1) else 0  # offset odd columns for tessellation
        for row in range(n_rows):
            cy = miny + row * row_spacing + row_offset
            polygons.append(_make_hexagon(cx, cy, R))

    hex_gdf = gpd.GeoDataFrame(
        {"hex_int_id": np.arange(1, len(polygons) + 1)},  # 0 reserved for "no hexagon"
        geometry=polygons,
        crs=target_crs_epsg,
    )
    print(f"Generated {len(hex_gdf):,} hexagons (~{hex_area_km2} km^2 each)")

    # Optional: clip to boundary
    if boundary_path:
        hex_gdf = gpd.overlay(hex_gdf, boundary_gdf[["geometry"]], how="intersection")
        print(f"Clipped to boundary: {len(hex_gdf):,} hexagons remain")

    # Reproject hexagons to raster CRS for rasterization
    hex_gdf_raster_crs = hex_gdf.to_crs(raster_crs)

    # -------------------------------------------------------------
    # 3. RASTERIZE HEXAGON IDs ONTO THE RASTER GRID
    # -------------------------------------------------------------
    shapes = zip(hex_gdf_raster_crs.geometry, hex_gdf_raster_crs.hex_int_id)
    hex_id_raster = rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.int32,
        all_touched=False,
    )

    # -------------------------------------------------------------
    # 4. VECTORIZED PER-HEXAGON PROPORTION (numpy.bincount, no looping)
    # -------------------------------------------------------------
    flat_ids = hex_id_raster.ravel()
    flat_binary = binary.ravel()
    flat_valid = valid_mask.ravel()

    flat_ids_valid = np.where(flat_valid, flat_ids, 0)

    max_id = flat_ids_valid.max()
    total_counts = np.bincount(flat_ids_valid, minlength=max_id + 1)
    positive_counts = np.bincount(flat_ids_valid, weights=flat_binary, minlength=max_id + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        proportion = np.where(total_counts > 0, positive_counts / total_counts, np.nan)

    hex_gdf["cell_count"] = hex_gdf["hex_int_id"].apply(lambda i: int(total_counts[i]) if i <= max_id else 0)
    hex_gdf["prop_disco"] = hex_gdf["hex_int_id"].apply(lambda i: proportion[i] if i <= max_id else np.nan)

    # Drop hexagons with no overlapping raster cells
    hex_gdf = hex_gdf[hex_gdf["cell_count"] > 0].copy()

    # -------------------------------------------------------------
    # 5. SAVE
    # -------------------------------------------------------------
    driver = "ESRI Shapefile" if str(out_hex_path).lower().endswith(".shp") else None
    if driver:
        hex_gdf.to_file(out_hex_path, driver=driver)
    else:
        hex_gdf.to_file(out_hex_path)

    print(f"Done. Saved {len(hex_gdf):,} hexagons with 'prop_disco' field to: {out_hex_path}")

    return hex_gdf


if __name__ == "__main__":
    pass

