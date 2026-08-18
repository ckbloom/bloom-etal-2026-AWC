# Extreme drought stress provides an ecological benchmark for high-resolution available water capacity maps of Swiss forest soils

Colin Bloom, Katrin Meusburger, Lorenz Walthert, and Andri Baltensweiler

This repository contains the analysis code for the manuscript above. It
computes available water capacity (AWC) for Swiss forest soils from
high-resolution soil property maps using ten pedotransfer functions
(PTFs), compares the resulting AWC maps against remotely sensed canopy
discoloration during the 2018 central European drought, and fits/evaluates
drought-susceptibility logistic regression models from AWC and seasonal
climate covariates.

`notebooks/AWC.ipynb` is the single entry point that reproduces the figures
and tables in the paper; everything it calls lives under `src/`.

## Repository structure

```
SOILS_2026/
├── notebooks/
│   └── AWC.ipynb              # Full analysis workflow - run this
├── src/
│   ├── retrieve_envidat_data.py   # Downloads soil property maps from EnviDat
│   ├── discoch_overview.py        # Hexagon-binned discoloration summary (Fig. 1)
│   ├── utils.py                   # Soil-cube loading, MvG water retention, AWC export
│   ├── ptfs/                      # One module per pedotransfer function (Table 1)
│   ├── combine_awc.py             # Multi-PTF mean/SD AWC rasters
│   ├── compare_awc_distributions.py  # AWC distribution boxplots (Fig. 2)
│   ├── national_stats.py          # National discoloration-vs-AWC binned statistics
│   ├── plot_ptf_averages.py       # Multi-PTF-average discoloration plots (Figs. 3)
│   ├── plot_ptf_individual.py     # Per-PTF discoloration plots (Fig. 4)
│   ├── dsi.py                     # Drought Susceptibility Index (DSI) logistic
│   │                                 regression pipeline, nested spatial CV (Figs. 5-6, Tables S1-S4)
│   └── model_application.py       # Applies a fitted DSI model across years (Figs. A2-A3)
├── data/                       # Inputs go here (not committed - see data/README.md)
├── output/                     # Pipeline outputs are written here
├── environment.yml
└── LICENSE
```

## Setup

```bash
conda env create -f environment.yml
conda activate awc-disco
```

The Python environment covers everything except one pedotransfer function
(Szabo et al., 2021), which is only available as an R implementation. To run
it you additionally need R with the `euptf2` and `terra` packages:

```r
install.packages("terra")
devtools::install_github("tkdweber/euptf2")
```

See `src/ptfs/szabo.R` for usage - it is a standalone script (not called from
the notebook) that writes `awc_1m_szabo.tif` / `awc_2m_szabo.tif` alongside
the other PTF outputs, so run it once before the cells in the notebook that
consume the `szabo` dataset.

## Data

`notebooks/AWC.ipynb` downloads the Swiss soil property maps automatically.
Discoloration maps and meteorological covariates are restricted-access and
must be requested from the corresponding author / Meteotest. See
[`data/README.md`](data/README.md) for the full breakdown of expected inputs
and their sources.

## Running the analysis

1. Activate the environment and launch Jupyter from `notebooks/`:
   ```bash
   cd notebooks
   jupyter lab AWC.ipynb
   ```
2. Run the "Environment Setup" and "User defined parameters" cells first -
   update `climate_data_dir` at the top of the parameters cell to point at
   your local climate rasters (these are not distributed with the
   repository).
3. Cells are largely sequential; the notebook's markdown headers note where
   later sections depend on the pickled/rasterized output of earlier ones
   (e.g. PTFs must be applied before distribution or DSI comparisons run).
4. All outputs (AWC rasters, pickled model fits, figures) are written under
   `output/`.

## Pedotransfer functions

| PTF | Reference | Module |
|---|---|---|
| Saxton | Saxton & Rawls (2006) | `src/ptfs/saxton_and_rawls.py` |
| Tóth | Tóth et al. (2015) | `src/ptfs/toth.py` |
| Zhang | Zhang & Schaap (2017), via `rosetta-soil` | `src/ptfs/zhang_and_schaap.py` |
| Wösten | Wösten et al. (1999) | `src/ptfs/wosten.py` |
| Weynants | Weynants et al. (2009) | `src/ptfs/weynants.py` |
| Puhlmann | Puhlmann & von Wilpert (2011) | `src/ptfs/puhlmann.py` |
| Teepe | Teepe et al. (2003) | `src/ptfs/teepe.py` |
| Schoch | Schoch et al. (2025) | `src/ptfs/schoch.py` |
| Wessolek | Wessolek et al. (2009) | `src/ptfs/wessolek.py` |
| Szabó | Szabó et al. (2021) | `src/ptfs/szabo.R` (R only) |

## Citation

If you use this code, please cite the manuscript above. Software citation
details will be added on publication.

## License

MIT - see [LICENSE](LICENSE).
