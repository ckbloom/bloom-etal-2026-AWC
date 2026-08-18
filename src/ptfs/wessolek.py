"""
wessolek.py
===========
MvG parameter computation using the pedotransfer functions of:
    Wessolek, G., Kaupenjohann, M., Renger, M. (2009).

Approach
--------
No closed-form equations are evaluated; MvG parameters are assigned via
a lookup table indexed by Ka5 texture class code.

where:
    clay : clay content        (%)          used for Ka5 classification
    silt : silt content        (%)          used for Ka5 classification
    Ka5  : Ka5 texture class   (-)          integer code, 0 = Unknown
    θs   : saturated water content  (cm³ cm⁻³)  from lookup table
    θr   : residual water content   (cm³ cm⁻³)  from lookup table
    α    : van Genuchten α          (hPa⁻¹)     from lookup table
    n    : van Genuchten n          (-)         from lookup table
"""


import numpy as np
import pandas as pd
import xarray as xr

# Integer codes for every Ka5 class.
# 0 is reserved for "Unknown" — pixels that match no rule stay 0.
KA5_CODES = {
    "Tt": 1, "Ss": 2, "Uu": 3, "Tu4": 4,
    "Ut2": 5, "Ut3": 6, "Ut4": 7, "Tu2": 8,
    "Ts2": 9, "Ts3": 10, "Ts4": 11, "St3": 12,
    "St2": 13, "Su2": 14, "Su3": 15, "Su4": 16,
    "Us": 17, "Uls": 18, "Lu": 19, "Tu3": 20,
    "Slu": 21, "Ls2": 22, "Lt2": 23, "Lt3": 24,
    "Sl2": 25, "Sl3": 26, "Sl4": 27, "Ls3": 28,
    "Ls4": 29, "Lts": 30, "Tl": 31,
    "fS": 32, "fSms": 33, "fSgs": 34, "mS": 35,
    "mSfs": 36, "mSgs": 37, "gS": 38
}

# Reverse map for human-readable output if needed
KA5_NAMES = {v: k for k, v in KA5_CODES.items()}
KA5_NAMES[0] = "Unknown"


def get_ka5_int(ds):
    """
    Fully vectorized Ka5 classification returning int16 codes.

    Encoding as integers (rather than object strings) keeps all arrays
    in numeric dtype so Dask can parallelize every chunk independently.
    Code 0 = Unknown (pixels that match no classification rule).

    Order of xr.where calls matters: more-specific rules must come last
    because each call overwrites the previous result where its condition
    is True.
    """
    silt = ds['silt']
    clay = ds['clay']

    # Start everything at 0 (Unknown)
    res = xr.zeros_like(clay, dtype=np.int16)

    res = xr.where(clay >= 65, KA5_CODES["Tt"], res)
    res = xr.where((clay < 5) & (silt < 10), KA5_CODES["Ss"], res)
    res = xr.where((clay < 8) & (silt >= 80), KA5_CODES["Uu"], res)
    res = xr.where((clay >= 25) & (silt >= 65), KA5_CODES["Tu4"], res)
    res = xr.where((clay >= 8) & (clay < 12) & (silt >= 65), KA5_CODES["Ut2"], res)
    res = xr.where((clay >= 12) & (clay < 17) & (silt >= 65), KA5_CODES["Ut3"], res)
    res = xr.where((clay >= 17) & (clay < 25) & (silt >= 65), KA5_CODES["Ut4"], res)
    res = xr.where((clay < 65) & (clay >= 45) & (silt >= 30), KA5_CODES["Tu2"], res)
    res = xr.where((clay < 65) & (clay >= 45) & (silt < 15), KA5_CODES["Ts2"], res)
    res = xr.where((clay >= 35) & (clay < 45) & (silt < 15), KA5_CODES["Ts3"], res)
    res = xr.where((clay >= 25) & (clay < 35) & (silt < 15), KA5_CODES["Ts4"], res)
    res = xr.where((clay >= 17) & (clay < 25) & (silt < 15), KA5_CODES["St3"], res)
    res = xr.where((clay >= 5) & (clay < 17) & (silt < 10), KA5_CODES["St2"], res)
    res = xr.where((clay < 5) & (silt >= 10) & (silt < 25), KA5_CODES["Su2"], res)
    res = xr.where((clay < 8) & (silt >= 25) & (silt < 40), KA5_CODES["Su3"], res)
    res = xr.where((clay < 8) & (silt >= 40) & (silt < 50), KA5_CODES["Su4"], res)
    res = xr.where((clay < 8) & (silt >= 50) & (silt < 80), KA5_CODES["Us"], res)
    res = xr.where((clay >= 8) & (clay < 17) & (silt >= 50) & (silt < 65), KA5_CODES["Uls"], res)
    res = xr.where((clay >= 17) & (clay < 30) & (silt >= 50) & (silt < 65), KA5_CODES["Lu"], res)
    res = xr.where((clay >= 30) & (clay < 45) & (silt >= 50) & (silt < 65), KA5_CODES["Tu3"], res)
    res = xr.where((clay >= 8) & (clay < 17) & (silt >= 40) & (silt < 50), KA5_CODES["Slu"], res)
    res = xr.where((clay >= 17) & (clay < 25) & (silt >= 40) & (silt < 50), KA5_CODES["Ls2"], res)
    res = xr.where((clay >= 25) & (clay < 35) & (silt >= 30) & (silt < 50), KA5_CODES["Lt2"], res)
    res = xr.where((clay >= 35) & (clay < 45) & (silt >= 30) & (silt < 50), KA5_CODES["Lt3"], res)
    res = xr.where((clay >= 5) & (clay < 8) & (silt >= 10) & (silt < 25), KA5_CODES["Sl2"], res)
    res = xr.where((clay >= 8) & (clay < 12) & (silt >= 10) & (silt < 40), KA5_CODES["Sl3"], res)
    res = xr.where((clay >= 12) & (clay < 17) & (silt >= 10) & (silt < 40), KA5_CODES["Sl4"], res)
    res = xr.where((clay >= 17) & (clay < 25) & (silt >= 30) & (silt < 40), KA5_CODES["Ls3"], res)
    res = xr.where((clay >= 17) & (clay < 25) & (silt >= 15) & (silt < 30), KA5_CODES["Ls4"], res)
    res = xr.where((clay >= 25) & (clay < 45) & (silt >= 15) & (silt < 30), KA5_CODES["Lts"], res)
    res = xr.where((clay >= 45) & (clay < 65) & (silt >= 15) & (silt < 30), KA5_CODES["Tl"], res)

    return res.astype(np.int16)


def apply_wessolek(ds, csv_path):
    """
    Wessolek et al. (2009) PTF — Ka5 lookup table.

    Expected variables in ds:
        clay  (%)  : clay content
        silt  (%)  : silt content
        Ka5   (-)  : integer-coded Ka5 texture class

    New variables added to ds:
        ths, thr, alpha, n   — MvG parameters
    """
    wesso_tab = pd.read_csv(csv_path, delimiter=',').set_index('texture')
    n_codes = max(KA5_CODES.values()) + 1

    luts = {}
    for param in ['alpha', 'n', 'ths', 'thr']:
        lut = np.full(n_codes, np.nan, dtype=np.float32)
        for name, code in KA5_CODES.items():
            if name in wesso_tab.index:
                lut[code] = wesso_tab.loc[name, param]
        luts[param] = lut

    ka5 = ds['Ka5'].values  # numpy array, shape (depth, y, x)

    for param, lut in luts.items():
        arr = lut[ka5]       # plain numpy fancy indexing — no lambda, no apply_ufunc
        ds[param] = xr.DataArray(
            arr.astype(np.float32),
            coords=ds['Ka5'].coords,
            dims=ds['Ka5'].dims,
        )

    return ds


if __name__ == "__main__":
    pass
