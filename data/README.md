# Data

This folder is where `notebooks/AWC.ipynb` expects its raster/vector inputs at
run time. Nothing large is committed to the repository; the pipeline downloads
or expects the following, organized as subfolders here:

| Path (relative to `data/`) | Contents | Source |
|---|---|---|
| `swiss_soil_maps/` | 25 m Swiss forest soil property maps (clay, sand, gravel, bulk density, SOC, rooting depth) by depth interval | Downloaded automatically by `src/retrieve_envidat_data.py` (EnviDat package `soil-property-maps-for-the-swiss-forest`, Baltensweiler et al., 2021) |
| `swiss_disco_maps/` | Discoloration probability rasters (`Disco_CH_<date>.tif`) | Available on request from the corresponding author |
| `swiss_climate_maps/` | All climate covariate rasters used in the DSI fits, one file per year/variable: `pr_seasonal_<year>_DJFMAM.tif` (winter/spring precipitation), `pr_seasonal_<year>_JJA.tif` (summer precipitation), `tas_seasonal_<year>_JJA.tif` (mean summer temperature), `tasmax_seasonal_<year>_JJA.tif` (max summer temperature), `vpd_<year>_JJA.tif` (mean summer vapor pressure deficit) | Meteorological data must be requested from Meteotest; the corresponding author can provide the specific derived rasters used here to peer reviewers on request |

`wessolek_MVG_tab10.csv` (included in this folder) holds the published Mualem-
van Genuchten parameter table from Wessolek et al. used directly by the
Wessolek PTF (`src/ptfs/wessolek.py`) - it is a small, fixed lookup table.

See the top of `notebooks/AWC.ipynb` ("User defined parameters") for the exact
directory variables the notebook uses, and the main [README](../README.md)
for how each pedotransfer function's outputs feed into the rest of the
analysis.
