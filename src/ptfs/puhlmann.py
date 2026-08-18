"""
puhlmann.py
===========
MvG parameter computation using the pedotransfer functions of:
    Puhlmann, H., von Wilpert, K. (2011).

Equations (Table 2)
--------------------
    θs      =  0.015362·SOC^0.5 - 0.2513·BD - 0.026836·ln(Cl+1) - 0.0055404·Sa^0.5 + 0.8648
    ln(α)   = -1.187·BD² - 0.031899·S - 0.58805·ln(C+0.1) - 0.00032963·U² - 0.016267·U·BD + 2.021
    ln(n−1) =  0.0003758·S² + 0.004751·U + 0.017826·U/BD - 2.9804

where:
    BD   : bulk density        (g cm⁻³)
    SOC   : organic carbon      (%)
    Cl   : clay content        (%)
    S    : sand content        (%)
    U    : silt content        (%)   = 100 − Sand − Clay
    α    : van Genuchten α     (hPa⁻¹)  recovered via exp(ln α)
    n    : van Genuchten n     (-)      recovered via exp(ln(n−1)) + 1

Notes
-----
Estimated θs, α, n are evaluated at FC and PWP via the standard van
Genuchten equation. This PTF set does not predict θr; it is fixed at 0
(see apply_puhlmann).
"""

import numpy as np
import xarray as xr


def ptf_ths(SOC, BD, Cl, Sa):
    """
    Saturated water content / θs (cm³ cm⁻³):
        θs = 0.015362·SOC^0.5 - 0.2513·BD - 0.026836·ln(Cl+1) - 0.0055404·Sa^0.5 + 0.8648
    R² = 0.621

    """
    return (
            0.015362 * np.sqrt(SOC)
            - 0.2513 * BD
            - 0.026836 * np.log(Cl + 1.0)
            - 0.0055404 * np.sqrt(Sa)
            + 0.8648
    ).astype(np.float32)


def ptf_alpha(BD, Sa, SOC, Sl):
    """
    Van Genuchten α (hPa⁻¹) via ln(α):
        ln(α) = -1.187·db² - 0.031899·S - 0.58805·ln(C+0.1) - 0.00032963·U² - 0.016267·U·db + 2.021
    R² = 0.316

    Returns α in hPa⁻¹.
    """
    ln_alpha = (
            -1.187 * BD ** 2
            - 0.031899 * Sa
            - 0.58805 * np.log(SOC + 0.1)
            - 0.00032963 * Sl ** 2
            - 0.016267 * Sl * BD
            + 2.021
    )
    return np.exp(ln_alpha).astype(np.float32)


def ptf_n(Sa, Sl, BD):
    """
    Van Genuchten n (−) via ln(n−1):
        ln(n−1) = 0.0003758·S² + 0.004751·U + 0.017826·U/db - 2.9804
    R² = 0.462

    Returns n (always > 1 by construction).
    """
    ln_n_minus1 = (
            0.0003758 * Sa ** 2
            + 0.004751 * Sl
            + 0.017826 * Sl / np.where(BD > 0, BD, np.nan)
            - 2.9804
    )
    return (np.exp(ln_n_minus1) + 1.0).astype(np.float32)


def apply_puhlmann(ds):
    """
    Puhlmann and von Wilpert 2011.

    Expected variables in ds:
        sand  (%)  : sand content
        clay  (%)  : clay content
        soc   (%)  : soil organic carbon
        trd  (g cm⁻³)  : bulk density

    New variables added to ds:
        ths, alpha, n   — MvG parameters
    """
    # Derive Silt content
    ds["silt"] = (100.0 - ds["sand"] - ds["clay"]).clip(0, 100).astype(np.float32)

    Sa = ds["sand"].astype(np.float32)
    Cl = ds["clay"].astype(np.float32)
    Sl = ds["silt"].astype(np.float32)
    SOC = ds["soc"].astype(np.float32)
    BD = ds["trd"].astype(np.float32)

    # --- MvG parameters via PTF -------------------------------------------
    ths = ptf_ths(SOC, BD, Cl, Sa)
    alpha = ptf_alpha(BD, Sa, SOC, Sl)
    n = ptf_n(Sa, Sl, BD)
    # Fixed residual water content (PTF set doesn't predict θr; use 0)
    thr = xr.zeros_like(ths)

    ds["ths"] = ths
    ds["thr"] = thr
    ds["alpha"] = alpha
    ds["n"] = n

    return ds


if __name__ == "__main__":
    pass