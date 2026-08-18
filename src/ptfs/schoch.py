"""
schoch.py
=========
MvG parameter computation using the pedotransfer functions of:
    Schoch, J., Nussbaum, M., Walthert, L., Carminati, A., Lehmann, P. (2025).

Equations (S1a - S1f)
----------------------
    θsat          =  0.919 − 0.355·BD + 0.000647·Clay + 0.000434·Sand − 0.00229·ln(OC)
    ln(θr + 0.2)  = −1.56  − 0.217·BD + 0.0106·Clay  + 0.00490·Sand
    ln(α)         = −3.84  − 1.32·BD  + 0.0237·Clay  + 0.0353·Sand  + 0.210·ln(OC)
    n⁻⁴           =  0.240 + 0.220·BD + 0.00291·Clay − 0.00305·Sand + 0.0329·ln(OC)
    √τ + 3        =  2.8469 − 0.229·BD − 0.0106·Clay − 0.00428·Sand + 0.0685·ln(OC)
    ln(Ksat)      =  7.45  − 3.16·BD  − 0.0364·Clay  + 0.0414·Sand  + 0.327·ln(OC)

where:
    BD   : bulk density        (g cm⁻³)
    OC   : organic carbon      (%)
    Clay : clay content        (%)
    Sand : sand content        (%)
    α    : van Genuchten α     (cm⁻¹)  recovered via exp(ln α)
    n    : van Genuchten n     (-)     recovered via (n⁻⁴)^(−1/4)
    τ    : tortuosity          (-)     recovered via (√τ+3 − 3)²
    Ksat : sat. hyd. cond.     (cm/d)  recovered via exp(ln Ksat)

Notes
-----
If predicted θr < 0, the soil lies outside the PTF domain.
"""
import numpy as np
import xarray as xr


def schoch_ths(bd, cl, sa, oc):
    """
    θsat (cm³ cm⁻³) — saturated water content.
    θsat = 0.919 − 0.355·BD + 0.000647·Clay + 0.000434·Sand − 0.00229·ln(OC)
    """
    return (
            0.919
            - 0.355 * bd
            + 0.000647 * cl
            + 0.000434 * sa
            - 0.00229 * np.log(oc)
    ).astype(np.float32)


def schoch_thr(bd, cl, sa):
    """
    θr (cm³ cm⁻³) — residual water content.
    ln(θr + 0.2) = −1.56 − 0.217·BD + 0.0106·Clay + 0.00490·Sand
    """
    return (np.exp(-1.56 - 0.217 * bd + 0.0106 * cl + 0.00490 * sa) - 0.2).astype(np.float32)


def schoch_alpha(bd, cl, sa, oc):
    """
    α (cm⁻¹) via ln(α).
    ln(α) = −3.84 − 1.32·BD + 0.0237·Clay + 0.0353·Sand + 0.210·ln(OC)
    """
    return np.exp(-3.84 - 1.32 * bd + 0.0237 * cl + 0.0353 * sa + 0.210 * np.log(oc)).astype(np.float32)


def schoch_n(bd, cl, sa, oc):
    """
    n (−) via n⁻⁴.
    n⁻⁴ = 0.240 + 0.220·BD + 0.00291·Clay − 0.00305·Sand + 0.0329·ln(OC)
    """
    n_inv4 = 0.240 + 0.220 * bd + 0.00291 * cl - 0.00305 * sa + 0.0329 * np.log(oc)
    return (n_inv4 ** -0.25).astype(np.float32)


def schoch_tau(bd, cl, sa, oc):
    """
    τ (−) via √τ + 3.
    √τ + 3 = 2.8469 − 0.229·BD − 0.0106·Clay − 0.00428·Sand + 0.0685·ln(OC)
    """
    expr = 2.8469 - 0.229 * bd - 0.0106 * cl - 0.00428 * sa + 0.0685 * np.log(oc)
    return ((expr - 3.0) ** 2).astype(np.float32)


def schoch_ksat(bd, cl, sa, oc):
    """
    Ksat (cm d⁻¹) via ln(Ksat).
    ln(Ksat) = 7.45 − 3.16·BD − 0.0364·Clay + 0.0414·Sand + 0.327·ln(OC)
    """
    return np.exp(7.45 - 3.16 * bd - 0.0364 * cl + 0.0414 * sa + 0.327 * np.log(oc)).astype(np.float32)


def apply_schoch(ds):
    """
    Schoch et al. (2025) PTF — full model (S1a-S1f).

    Expected variables in ds:
        sand  (%)         : sand content
        clay  (%)         : clay content
        soc   (%)         : soil organic carbon
        trd   (g cm⁻³)    : bulk density

    New variables added to ds:
        ths, thr, alpha, n, tau, ksat   — MvG parameters

    All parameters are set to NaN where θr < 0, indicating the soil
    lies outside the PTF domain.
    """
    sa = ds["sand"].astype(np.float32)
    cl = ds["clay"].astype(np.float32)
    bd = ds["trd"].astype(np.float32)
    oc = ds["soc"].astype(np.float32)

    thr = schoch_thr(bd, cl, sa)
    valid = thr >= 0

    def mask(arr):
        return xr.where(valid, arr, np.nan).astype(np.float32)

    ds["ths"] = mask(schoch_ths(bd, cl, sa, oc))
    ds["thr"] = mask(thr)
    ds["alpha"] = mask(schoch_alpha(bd, cl, sa, oc))
    ds["n"] = mask(schoch_n(bd, cl, sa, oc))
    ds["tau"] = mask(schoch_tau(bd, cl, sa, oc))
    ds["ksat"] = mask(schoch_ksat(bd, cl, sa, oc))
    return ds


if __name__ == "__main__":
    pass
