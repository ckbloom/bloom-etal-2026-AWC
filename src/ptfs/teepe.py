"""
teepe.py
========
AWC and MvG parameter computation using the pedotransfer functions of:
    Teepe, R., Dilling, H., Beese, F. (2003).

Equations (direct AWC, R² = 38.1%)
------------------------------------
    AWC = 0.4172514
          + 0.1641025 · BD
          − 0.0012394 · Clay
          + 0.00061115 · Silt

Equations (MvG parameters)
----------------------------
    θs       = 0.9786 - BD · 0.36686
    θr       = 0.0
    ln(α)    = 55.576 - 4.433 · BD - 0.002 · Silt² - 0.470 · Clay
               - 0.066 · (Sand/BD) - 3.683 · Sand^0.5 - 0.0359 · (Silt/BD)
               - 0.0016 · Sand² - 3.6916 · Silt^0.5 + 1.8643 · ln(Sand) + 1.575 · ln(Silt)
    ln(n-1)  = -2.8497 + Sand² · 0.00027395 + 0.01637 · Silt

where:
    AWC  : available water capacity  (cm³ cm⁻³)  at -60 to -16000 kPa
    BD   : bulk density              (g cm⁻³)
    Clay : clay content              (%)
    Silt : silt content              (%)
    Sand : sand content              (%)
    α    : van Genuchten α           (cm⁻¹)  recovered via exp(ln α)
    n    : van Genuchten n           (-)     recovered via exp(ln(n-1)) + 1

Notes
-----
apply_teepe(awc_direct=True) uses the direct AWC regression; otherwise
it computes the MvG parameters (θr fixed at 0, not predicted by this
PTF set).
"""

import numpy as np

def teepe_ths(bd):
    """
    θs (cm³ cm⁻³) — saturated water content.
    θs = 0.9786 - BD * 0.36686
    """
    return (0.9786 - bd * 0.36686).astype(np.float32)


def teepe_thr(bd):
    """
    θr (cm³ cm⁻³) — residual water content.
    Assumed zero.
    """
    # Returning zeros matching the shape of the input
    return (bd * 0.0).astype(np.float32)


def teepe_alpha(sa, sl, cl, bd):
    """
    α (cm⁻¹) via ln(α).
    """

    ln_alpha = (
        55.576
        - 4.433 * bd
        - 0.002 * (sl ** 2)
        - 0.470 * cl
        - 0.066 * (sa / bd)
        - 3.683 * (sa ** 0.5)
        - 0.0359 * (sl / bd)
        - 0.0016 * (sa ** 2)
        - 3.6916 * (sl ** 0.5)
        + 1.8643 * np.log(sa)
        + 1.575 * np.log(sl)
    )
    return np.exp(ln_alpha).astype(np.float32)


def teepe_n(sa, sl):
    """
    n (−) via ln(n-1).
    ln(n-1) = -2.8497 + Sand**2 * 0.00027395 + 0.01637 * Silt
    """
    ln_n_minus_1 = (
        -2.8497
        + (sa ** 2) * 0.00027395
        + 0.01637 * sl
    )
    return (np.exp(ln_n_minus_1) + 1.0).astype(np.float32)


def apply_teepe(ds, awc_direct=False):
    """
    Compute Teepe et al. (2003) moisture parameters and AWC for all depth layers.

    Inputs expected in ds:
        sand  (%)  : sand content
        clay  (%)  : clay content
        trd   ()   : bulk density (g cm⁻³)

    Adds to ds:
        silt    (%)      : computed silt content
        awc_vol (cm³/cm³): plant available water
        ths     (cm³/cm³): saturated water content
        thr     (cm³/cm³): residual water content
        alpha   (cm⁻¹)   : van Genuchten alpha parameter
        n       (-)      : van Genuchten n parameter
    """
    sa = ds["sand"].astype(np.float32)
    cl = ds["clay"].astype(np.float32)
    bd = ds["trd"].astype(np.float32)

    # Calculate silt first as it's needed for both AWC and MvG parameters
    sl = (100.0 - sa - cl).clip(0, 100).astype(np.float32)
    ds["silt"] = sl

    if awc_direct:
        # Compute AWC
        ds['awc_vol'] = (
                0.4172514
                + 0.1641025 * bd
                - 0.0012394 * cl
                + 0.00061115 * sl
        )

    else:
        # Compute MvG Parameters
        ds["ths"] = teepe_ths(bd)
        ds["thr"] = teepe_thr(bd)
        ds["alpha"] = teepe_alpha(sa, sl, cl, bd)
        ds["n"] = teepe_n(sa, sl)

    return ds


if __name__ == "__main__":
    pass
