# =============================================================================
# szabo.R
# ============
# MvG computation using the pedotransfer functions of:
#   Szabó, B., Weynants, M. and Weber, T. K. D. (2021)
#
# Strategy (as recommended by the authors):
#   - Predict FC  (water content at pF 2.5, i.e. -330 cm / ~33 kPa) and
#     WP  (water content at pF 4.2, i.e. -15 000 cm / ~1500 kPa)
#     using point PTFs selected by which_PTF() for the available predictors.
#   - AWC (cm3/cm3) = FC - WP  (clipped to [0, 0.8])
#   - Layer AWC (mm) = AWC_vol * (1 - gravel_frac) * layer_thickness_mm
#   - Integrate over depth to give awc_2m, awc_1m, awc_mrd (mm).
#
# Input raster variable names expected on disk (same conventions as the
# Python script; adjust soil_names_de / depth_ranges as needed):
#   sand, clay, silt, trd (bulk density), corg-gehalt (OC), skelettgehalt (gravel)
#
# Outputs (EPSG:2056, LZW-compressed GeoTIFF, int16, nodata = -9999):
#   awc_2m_szabo.tif   – AWC integrated over 0–200 cm
#   awc_1m_szabo.tif   – AWC integrated over 0–100 cm
#
# Required packages:
#   euptf2   – devtools::install_github("tkdweber/euptf2")
#   terra    – install.packages("terra")
# =============================================================================

suppressPackageStartupMessages({
  library(euptf2)   # PTF predictions
  library(terra)    # raster I/O and reprojection
})

# ---------------------------------------------------------------------------
# 0.  Configuration  --------------------------------------------------------
# ---------------------------------------------------------------------------

soil_data_loc <- "../data/swiss_soil_maps"  # Adjust to match location

# Map from standardised names (used internally) to filename search keys
# Silt is derived as 100 - sand - clay, not loaded from disk.
soil_names_de <- list(
  sand   = "sand",
  clay   = "clay",
  gravel = "gravel",
  trd    = "density",
  soc    = "SOC"
)

soil_depth_name <- "depth"   # rooting-depth raster
depth_ranges    <- c("0_5", "5_15", "15_30", "30_60", "60_100", "100_200")

nodata_value <- -9999L
output_dir   <- "../awc"  # Adjust with desired output path
suffix       <- "_szabo"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Helper: find the single .tif file matching a search key + depth string
# ---------------------------------------------------------------------------
find_tif <- function(dir, key, depth) {
  pattern <- file.path(dir, paste0("*", key, "*", depth, "*.tif"))
  hits <- Sys.glob(pattern)
  if (length(hits) == 0L) {
    warning(sprintf("No file found for key='%s' depth='%s' in %s", key, depth, dir))
    return(NULL)
  }
  hits[1L]
}

# ---------------------------------------------------------------------------
# 1.  Load soil rasters into a named list of SpatRaster stacks  -------------
#     Each element: a SpatRaster with one layer per depth slice
# ---------------------------------------------------------------------------
message("Loading soil data ...")

depth_upper <- as.integer(sub("_.*", "", depth_ranges))
depth_lower <- as.integer(sub(".*_", "", depth_ranges))
thick_mm    <- (depth_lower - depth_upper) * 10L   # thickness in mm

vars <- names(soil_names_de)
soil_stacks <- setNames(vector("list", length(vars)), vars)

for (v in vars) {
  key   <- soil_names_de[[v]]
  layers <- lapply(depth_ranges, function(dr) {
    f <- find_tif(soil_data_loc, key, dr)
    if (is.null(f)) return(NULL)
    rast(f)
  })
  ok <- !sapply(layers, is.null)
  if (!any(ok)) stop(sprintf("Variable '%s' not found for any depth layer.", v))
  soil_stacks[[v]] <- rast(layers[ok])
  names(soil_stacks[[v]]) <- depth_ranges[ok]
}

# Use sand as the reference grid
ref_rast <- soil_stacks[["sand"]][[1]]


# ---------------------------------------------------------------------------
# 2.  Predict FC and WP per depth layer using euptf2 point PTFs  ------------
# ---------------------------------------------------------------------------
# euptf2 input column names:
#   DEPTH_M  – mean sampling depth (cm)  = (upper + lower) / 2
#   USSAND   – sand  (g/100g = %)
#   USSILT   – silt  (g/100g = %)
#   USCLAY   – clay  (g/100g = %)
#   OC       – organic carbon (g/100g = %)
#   BD       – bulk density (g/cm³)
#
# which_PTF() chooses the best available PTF for the provided predictor set.
# FC  → pF 2.5 / ~33 kPa  (closest to your 63 hPa / 64 cm FC definition)
# WP  → pF 4.2 / ~1500 kPa

message("Selecting best PTFs for FC and WP ...")

# Build a representative predictor frame (needed only to call which_PTF)
dummy_pred <- data.frame(
  DEPTH_M = 10,
  USSAND  = 40,
  USSILT  = 40,
  USCLAY  = 20,
  OC      = 1,
  BD      = 1.4
)

# which_PTF() returns a 1-row data.table whose single column is named after
# the target (e.g. column "FC" containing "PTF02"). Extract by position.
ptf_FC <- as.character(which_PTF(predictor = dummy_pred, target = "FC")[[1L]])
ptf_WP <- as.character(which_PTF(predictor = dummy_pred, target = "WP")[[1L]])

# Sanity check
stopifnot(length(ptf_FC) == 1L, grepl("^PTF[0-9]+$", ptf_FC))
stopifnot(length(ptf_WP) == 1L, grepl("^PTF[0-9]+$", ptf_WP))

message(sprintf("  FC  -> %s", ptf_FC))
message(sprintf("  WP  -> %s", ptf_WP))

# ---------------------------------------------------------------------------
# Per-layer prediction (vectorised over pixels)
# ---------------------------------------------------------------------------

predict_layer_awc <- function(depth_idx) {
  dr   <- depth_ranges[depth_idx]
  umid <- (depth_upper[depth_idx] + depth_lower[depth_idx]) / 2   # mean depth (cm)

  message(sprintf("  Processing depth layer %s (DEPTH_M = %.1f cm) ...", dr, umid))

  # Read rasters for this depth as matrices
  get_mat <- function(v) {
    r <- soil_stacks[[v]][[depth_idx]]
    as.vector(values(r, mat = FALSE))
  }

  sa  <- get_mat("sand")
  cl  <- get_mat("clay")
  sl  <- 100 - sa - cl       # silt derived; no raster needed
  oc  <- get_mat("soc")
  bd  <- get_mat("trd")
  gv  <- get_mat("gravel")

  n   <- length(sa)

  # Mask: pixels where all required vars are finite
  mask <- is.finite(sa) & is.finite(cl) & is.finite(bd)

  # Build predictor data.frame (only valid pixels)
  pred_df <- data.frame(
    DEPTH_M = umid,
    USSAND  = sa[mask],
    USSILT  = sl[mask],
    USCLAY  = cl[mask],
    OC      = ifelse(is.finite(oc[mask]), oc[mask], NA_real_),
    BD      = bd[mask]
  )

  # Predict FC and WP (mean predictions)
  res_fc <- euptfFun(ptf = ptf_FC,  predictor = pred_df, target = "FC",  query = "predictions")
  res_wp <- euptfFun(ptf = ptf_WP,  predictor = pred_df, target = "WP",  query = "predictions")

  # Extract predicted columns (euptfFun appends prediction columns to pred_df)
  fc_col  <- grep("^FC_",  names(res_fc), value = TRUE)[1L]
  wp_col  <- grep("^WP_",  names(res_wp), value = TRUE)[1L]

  fc_vals <- res_fc[[fc_col]]
  wp_vals <- res_wp[[wp_col]]

  # AWC volumetric (cm³/cm³), clipped
  awc_vol <- pmax(0, pmin(0.8, fc_vals - wp_vals))

  # Gravel correction and conversion to mm
  gv_frac    <- ifelse(is.finite(gv[mask]), gv[mask] / 100, 0)
  thick      <- thick_mm[depth_idx]
  awc_mm_vec <- awc_vol * (1 - gv_frac) * thick

  # Put values back into full-grid vectors
  awc_mm_full <- rep(NA_real_, n)
  awc_mm_full[mask] <- awc_mm_vec

  # Return as SpatRaster with same extent/crs as reference
  r_out <- ref_rast
  values(r_out) <- awc_mm_full
  r_out
}

message("Predicting FC and WP for each depth layer ...")
layer_awc_rasts <- lapply(seq_along(depth_ranges), predict_layer_awc)
awc_stack <- rast(layer_awc_rasts)   # depth as bands

# ---------------------------------------------------------------------------
# 3.  Depth integration  ----------------------------------------------------
# ---------------------------------------------------------------------------

message("Integrating AWC over depth ...")

# 3a. AWC over full 0–200 cm profile
awc_2m <- app(awc_stack, fun = function(x) sum(x, na.rm = FALSE))
# Use na.rm = TRUE when at least one layer is valid (mirrors min_count=1)
awc_2m <- app(awc_stack, fun = function(x) {
  if (all(is.na(x))) NA_real_ else sum(x, na.rm = TRUE)
})

# 3b. AWC over 0–100 cm profile (layers where upper depth < 100 cm)
is_1m <- depth_upper < 100
awc_1m <- app(awc_stack[[which(is_1m)]], fun = function(x) {
  if (all(is.na(x))) NA_real_ else sum(x, na.rm = TRUE)
})


# ---------------------------------------------------------------------------
# 4.  Export results  -------------------------------------------------------
# ---------------------------------------------------------------------------
message("Exporting results ...")

target_crs <- "EPSG:2056"

export_raster <- function(r, filename) {
  r_proj    <- project(r, target_crs, method = "bilinear")
  r_int     <- round(r_proj)
  r_int[is.na(r_int)] <- nodata_value
  datatype  <- "INT2S"
  out_path  <- file.path(output_dir, filename)
  writeRaster(
    r_int,
    out_path,
    overwrite    = TRUE,
    datatype     = datatype,
    gdal         = c("COMPRESS=LZW", "TILED=YES",
                     paste0("NODATA=", nodata_value)),
    NAflag       = nodata_value
  )
  message(sprintf("  Exported %s", filename))
}

export_raster(awc_2m,  paste0("awc_2m",  suffix, ".tif"))
export_raster(awc_1m,  paste0("awc_1m",  suffix, ".tif"))

message("Done.")