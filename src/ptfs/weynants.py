"""
weynants.py
===========
MvG parameter computation using the pedotransfer functions of:
    Weynants, M., Vereecken, H., Javaux, M. (2009).

Equations (Tables 3 and 6)
----------------------------
    θs      =  0.6355 + 0.0013·Cl - 0.1631·BD
    θr      =  0.0
    ln(α)   = -4.3003 - 0.0097·Cl + 0.0138·Sa - 0.0992·SOC
    ln(n-1) = -1.0846 - 0.0236·Cl - 0.0085·Sa + 0.0001·Sa²

where:
    BD   : bulk density        (g cm⁻³)
    SOC  : organic carbon      (%)
    Cl   : clay content        (%)
    Sa   : sand content        (%)
    α    : van Genuchten α     (cm⁻¹)  recovered via exp(ln α)
    n    : van Genuchten n     (-)     recovered via exp(ln(n-1)) + 1

Notes
-----
θr is fixed at 0 (not predicted by this PTF set).
"""

import numpy as np

def weynants_ths(bd, cl):
    """
    θs (cm³ cm⁻³) — saturated water content.
    θs = 0.6355 + 0.0013·Cl - 0.1631·BD
    """
    return (0.6355 + 0.0013 * cl - 0.1631 * bd).astype(np.float32)


def weynants_thr(cl):
    """
    θr (cm³ cm⁻³) — residual water content.
    Weynants assumes θr = 0.
    """
    # Returning zeros matching the shape of the input
    return (cl * 0.0).astype(np.float32)


def weynants_alpha(sa, soc, cl):
    """
    α (cm⁻¹) via ln(α).
    ln(α) = -4.3003 - 0.0097·Cl + 0.0138·Sa - 0.0992·C
    """
    ln_alpha = (
            -4.3003
            - 0.0097 * cl
            + 0.0138 * sa
            - 0.0992 * soc
    )
    return np.exp(ln_alpha).astype(np.float32)


def weynants_n(sa, cl):
    """
    n (−) via ln(n-1).
    ln(n-1) = -1.0846 - 0.0236·Cl - 0.0085·Sa + 0.0001·Sa²
    """
    ln_n_minus_1 = (
            -1.0846
            - 0.0236 * cl
            - 0.0085 * sa
            + 0.0001 * sa ** 2
    )
    return (np.exp(ln_n_minus_1) + 1.0).astype(np.float32)


def apply_weynants(ds):
    """
    Weynants (2009) PTF

    Expected variables in ds:
        sand  (%)  : sand content
        clay  (%)  : clay content
        soc   (%)  : soil organic carbon
        trd  (g cm⁻³)  : bulk density

    New variables added to ds:
        ths, thr, alpha, npar   — MvG parameters
    """
    sa = ds["sand"].astype(np.float32)
    cl = ds["clay"].astype(np.float32)
    bd = ds["trd"].astype(np.float32)
    soc = ds["soc"].astype(np.float32)

    # Parameters
    ths = weynants_ths(bd, cl)
    thr = weynants_thr(cl)
    alpha = weynants_alpha(sa, soc, cl)
    n = weynants_n(sa, cl)

    ds["ths"] = ths
    ds["thr"] = thr
    ds["alpha"] = alpha
    ds["n"] = n

    return ds


if __name__ == "__main__":
    pass
