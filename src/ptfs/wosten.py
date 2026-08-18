"""
wosten.py
=========
MvG parameter computation using the pedotransfer functions of:
    Wösten, J.H.M., Lilly, A., Nemes, A., Le Bas, C. (1999).

Equations (Table 2)
--------------------
    θs      =  0.7919 + 0.001691·Cl - 0.29619·BD - 1.491e-6·Sl²
               + 8.21e-5·OM² + 0.02427/Cl + 0.01113/Sl + 0.01472·ln(Sl)
               - 7.33e-5·OM·Cl - 0.000619·BD·Cl - 0.001183·BD·OM
               - 1.664e-4·TOP·Sl
    θr      =  0.0
    ln(α)   = -14.96 + 0.03135·Cl + 0.0351·Sl + 0.646·OM + 15.29·BD
               - 0.192·TOP - 4.671·BD² - 0.000781·Cl² - 0.00687·OM²
               + 0.0449/OM + 0.0663·ln(Sl) + 0.1482·ln(OM)
               - 0.04546·BD·Sl - 0.4852·BD·OM + 0.00673·TOP·Cl
    ln(n−1) = -25.23 - 0.02195·Cl + 0.0074·Sl - 0.1940·OM + 45.5·BD
               - 7.24·BD² + 3.658e-4·Cl² + 0.002885·OM² - 12.81/BD
               - 0.1524/Sl - 0.01958/OM - 0.2876·ln(Sl) - 0.0709·ln(OM)
               - 44.6·ln(BD) - 0.02264·BD·Cl + 0.0896·BD·OM + 0.00718·TOP·Cl

where:
    Cl   : clay content   (%)
    Sl   : silt content   (%)   = 100 − Sand − Clay
    BD   : bulk density   (g cm⁻³)
    OM   : organic matter (%)   = SOC% × van Bemmelen factor (1.72)
    TOP  : 1 for topsoil (depth < 30 cm), 0 for subsoil
    α    : van Genuchten α        (cm⁻¹)  recovered via exp(ln α)
    n    : van Genuchten n        (-)     recovered via exp(ln(n−1)) + 1

Notes
-----
Starred notation in the original paper (α*, n*, l*, Ks*) denotes the
natural log of the parameter — the equations above already give ln(α)
and ln(n−1) directly, matching this module's implementation. Ks and l
(Mualem connectivity/tortuosity exponent) are not implemented here.

OM is clipped per the paper's domain limits (≤18% if Cl>60%, else
≤12+0.1·Cl), and Cl, Sl, BD are floored to avoid division-by-zero and
log-domain errors. θr is fixed at 0 (not predicted by this PTF set).
"""

import numpy as np
import xarray as xr


VAN_BEMMELEN = 1.72  # SOC → SOM conversion factor (Wösten uses OM, not OC)


def apply_wosten(ds, van_bemmelen=VAN_BEMMELEN):
    """
    Wösten (1999) PTF

    Expected variables in ds:
        sand  (%)  : sand content
        clay  (%)  : clay content
        soc   (%)  : soil organic carbon
        trd  (g cm⁻³)  : bulk density

    New variables added to ds:
        ths, thr, alpha, n   — MvG parameters
    """

    Cl = ds["clay"].astype(np.float32)
    Sl = (100.0 - ds["sand"] - ds["clay"]).clip(0, 100).astype(np.float32)
    BD = ds["trd"].astype(np.float32)
    OM = (ds["soc"] * van_bemmelen).astype(np.float32)

    ds["silt"] = Sl

    top_vals = (ds.depth < 30).astype(np.float32)
    TOP = top_vals.broadcast_like(Cl)  # stays as DataArray, no .values

    # Clipping stays lazy
    OM = xr.where((Cl > 60) & (OM > 18), 18, OM)
    OM = xr.where((Cl <= 60) & (OM > 12 + 0.1 * Cl), 12 + 0.1 * Cl, OM)
    OM = OM.clip(0.1, None)
    BD = BD.clip(0.5, None)
    Cl = Cl.clip(0.5, None)
    Sl = Sl.clip(0.5, None)

    ths = (
            0.7919 + 0.001691 * Cl - 0.29619 * BD - 1.491e-6 * Sl ** 2
            + 8.21e-5 * OM ** 2 + 0.02427 / Cl + 0.01113 / Sl + 0.01472 * np.log(Sl)
            - 7.33e-5 * OM * Cl - 0.000619 * BD * Cl - 0.001183 * BD * OM
            - 1.664e-4 * TOP * Sl
    ).astype(np.float32)

    alpha = np.exp(
        -14.96 + 0.03135 * Cl + 0.0351 * Sl + 0.646 * OM + 15.29 * BD
        - 0.192 * TOP - 4.671 * BD ** 2 - 0.000781 * Cl ** 2 - 0.00687 * OM ** 2
        + 0.0449 / OM + 0.0663 * np.log(Sl) + 0.1482 * np.log(OM)
        - 0.04546 * BD * Sl - 0.4852 * BD * OM + 0.00673 * TOP * Cl
    ).astype(np.float32)

    n = (np.exp(
        -25.23 - 0.02195 * Cl + 0.0074 * Sl - 0.1940 * OM + 45.5 * BD
        - 7.24 * BD ** 2 + 3.658e-4 * Cl ** 2 + 0.002885 * OM ** 2 - 12.81 / BD
        - 0.1524 / Sl - 0.01958 / OM - 0.2876 * np.log(Sl) - 0.0709 * np.log(OM)
        - 44.6 * np.log(BD) - 0.02264 * BD * Cl + 0.0896 * BD * OM
        + 0.00718 * TOP * Cl
    ) + 1.0).astype(np.float32)

    ds["ths"] = ths
    ds["alpha"] = alpha
    ds["n"] = n
    ds["thr"] = xr.zeros_like(ths)

    return ds


if __name__ == "__main__":
    pass
