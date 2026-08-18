"""
saxton_and_rawls.py
===================
FC and WP computation using the pedotransfer functions of:
    Saxton, K.E., Rawls, W.J. (2006).

Equations (Table 1)
--------------------
Moisture regressions (Eqs. 1-5):
    theta_1500t  = -0.024*S + 0.487*C + 0.006*OM + 0.005*(S*OM)
                   - 0.013*(C*OM) + 0.068*(S*C) + 0.031
    theta_1500   = theta_1500t + (0.14*theta_1500t - 0.02)            [Eq. 1]

    theta_33t    = -0.251*S + 0.195*C + 0.011*OM + 0.006*(S*OM)
                   - 0.027*(C*OM) + 0.452*(S*C) + 0.299
    theta_33     = theta_33t + (1.283*theta_33t**2 - 0.374*theta_33t - 0.015) [Eq. 2]

    theta_S33t   = 0.278*S + 0.034*C + 0.022*OM - 0.018*(S*OM)
                   - 0.027*(C*OM) - 0.584*(S*C) + 0.078
    theta_S33    = theta_S33t + (0.636*theta_S33t - 0.107)            [Eq. 3]

    psi_et       = -21.67*S - 27.93*C - 81.97*theta_S33
                   + 71.12*(S*theta_S33) + 8.29*(C*theta_S33)
                   + 14.05*(S*C) + 27.16
    psi_e        = psi_et + (0.02*psi_et**2 - 0.113*psi_et - 0.70)   [Eq. 4]

    theta_S      = theta_33 + theta_S33 - 0.097*S + 0.043             [Eq. 5]

Density adjustment (Eqs. 6-10, optional, requires measured bulk density):
    rho_N        = (1 - theta_S) * 2.65                               [Eq. 6]
    rho_DF       = rho_N * DF                                         [Eq. 7]
    theta_S_DF   = 1 - (rho_DF / 2.65)                               [Eq. 8]
    theta_33_DF  = theta_33 - 0.2*(theta_S - theta_S_DF)             [Eq. 9]
    theta_S33_DF = theta_S_DF - theta_33_DF                          [Eq. 10]

where DF = rho_measured / rho_N  (density factor, typically 0.9-1.3)

Key outputs
-----------
    WP  (wilting point, 1500 kPa) = theta_1500
    FC  (field capacity,   33 kPa) = theta_33   (or theta_33_DF)
    SAT (saturation,        0 kPa) = theta_S    (or theta_S_DF)

Input unit conventions
-----------------------
    S  : sand as FRACTION (0-1)  -- divide % by 100
    C  : clay as FRACTION (0-1)  -- divide % by 100
    OM : organic matter as FRACTION (0-1)  -- divide % by 100
         If input is SOC (%), convert to OM first: OM% = SOC% * 1.72
         then divide by 100.
    BD : measured bulk density (g/cm3) -- only needed for density adjustment

All moisture outputs are volumetric fractions (cm3/cm3).
"""

import logging
import warnings
import numpy as np
import xarray as xr


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*angle from rectified to skew grid parameter lost.*",
    module="pyproj",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

VAN_BEMMELEN = 1.72  # SOC% -> OM% conversion factor


# ===========================================================================
# Saxton & Rawls (2006) equations
# All inputs S, C, OM are fractions (0-1), not percentages.
# ===========================================================================

def _theta_1500(S, C, OM):
    """Wilting point intermediate — Saxton & Rawls (2006) eq 1."""
    theta1500t = (-0.024 * S + 0.487 * C + 0.006 * OM
                  + 0.005 * (S * OM) - 0.013 * (C * OM)
                  + 0.068 * (S * C) + 0.031)
    return (theta1500t + (0.14 * theta1500t - 0.02)).astype(np.float32)


def _theta_33(S, C, OM):
    """Field capacity — Saxton & Rawls (2006) eq 2."""
    theta33t = (-0.251 * S + 0.195 * C + 0.011 * OM
                + 0.006 * (S * OM) - 0.027 * (C * OM)
                + 0.452 * (S * C) + 0.299)
    return (theta33t + (1.283 * theta33t ** 2 - 0.374 * theta33t - 0.015)).astype(np.float32)


def _theta_S33(S, C, OM):
    """Saturation-to-FC difference — Saxton & Rawls (2006) eq 3."""
    theta_s33t = (0.278 * S + 0.034 * C + 0.022 * OM
                  - 0.018 * (S * OM) - 0.027 * (C * OM)
                  - 0.584 * (S * C) + 0.078)
    return (theta_s33t + (0.636 * theta_s33t - 0.107)).astype(np.float32)


def _theta_S(th33, thS33, S):
    """Saturation — Saxton & Rawls (2006) eq 5."""
    return (th33 + thS33 - 0.097 * S + 0.043).astype(np.float32)


def _density_adjustment(thS, th33, DF):
    """
    Density correction — Saxton & Rawls (2006) eqs 6–10.
    DF  : density factor (0.9–1.3). Values outside this range → NaN.
    Returns corrected (thS_df, th33_df, thS33_df)
    """
    DF = xr.where((DF >= 0.9) & (DF <= 1.3), DF, np.nan)  # NaN outside calibrated range
    rho_N = (1.0 - thS) * 2.65  # eq 6
    rho_DF = rho_N * DF  # eq 7
    thS_df = (1.0 - (rho_DF / 2.65)).astype(np.float32)  # eq 8
    th33_df = (th33 - 0.2 * (thS - thS_df)).astype(np.float32)  # eq 9
    thS33_df = (thS_df - th33_df).astype(np.float32)  # eq 10
    return thS_df, th33_df, thS33_df


def apply_saxton_rawls(ds, density_factor=None, van_bemmelen=VAN_BEMMELEN):
    """
    Compute Saxton & Rawls (2006) moisture parameters and AWC for all depth layers.

    Inputs expected in ds:
        sand  (%)  : sand content
        clay  (%)  : clay content
        soc   (%)  : soil organic carbon, converted to OM via van Bemmelen factor

    Parameters:
        density_factor : float or None
            DF in Saxton & Rawls (2006) eq 7. Must be in [0.9, 1.3].
            If None, no density correction is applied.
        van_bemmelen : float
            SOC to OM conversion factor (default 1.72).

    Adds to ds:
        wp      : wilting point          (cm³/cm³)  — θ₁₅₀₀
        fc      : field capacity         (cm³/cm³)  — θ₃₃ or θ₃₃-DF
        sat     : saturation             (cm³/cm³)  — θS  or θS-DF
        awc_vol : plant available water  (cm³/cm³)  — fc - wp

    """
    # --- Convert inputs to fractions -----------------------------------------
    S = (ds["sand"] / 100.0).astype(np.float32)
    C = (ds["clay"] / 100.0).astype(np.float32)
    OM = (ds["soc"] * van_bemmelen / 100.0).astype(np.float32)

    # Guard: S + C must not exceed 1 (silt = 1 - S - C >= 0)
    excess = (S + C - 1.0).clip(0, None)
    S = S - excess / 2.0
    C = C - excess / 2.0

    # --- Base moisture regressions -------------------------------------------
    th1500 = _theta_1500(S, C, OM)
    th33 = _theta_33(S, C, OM)
    thS33 = _theta_S33(S, C, OM)
    thS = _theta_S(th33, thS33, S)

    # Ensure physical ordering: th1500 < th33 < thS
    th1500 = xr.where(th1500 >= th33, th33 * 0.95, th1500).astype(np.float32)
    th33 = xr.where(th33 >= thS, thS * 0.95, th33).astype(np.float32)

    logging.info("FC and WP calculated ...")

    # --- Optional density correction -----------------------------------------
    if density_factor is not None:
        if not (0.9 <= density_factor <= 1.3):
            raise ValueError(
                f"density_factor must be in [0.9, 1.3], got {density_factor}. "
                f"See Saxton & Rawls (2006) eq 7."
            )
        logging.info(f"  Applying density correction with DF={density_factor} ...")

        DF = xr.full_like(thS, fill_value=density_factor)
        thS_df, th33_df, _ = _density_adjustment(thS, th33, DF)

        thS = thS_df.astype(np.float32)
        th33 = th33_df.astype(np.float32)

        # Re-enforce ordering after density correction
        th1500 = xr.where(th1500 >= th33, th33 * 0.95, th1500).astype(np.float32)
        th33 = xr.where(th33 >= thS, thS * 0.95, th33).astype(np.float32)

        logging.info(f"  Density correction applied (DF={density_factor})")
    else:
        logging.info("  No density correction (DF not provided, using textural estimates only)")

    # --- Store outputs -------------------------------------------------------
    ds["wp"] = th1500
    ds["fc"] = th33
    ds["sat"] = thS
    ds["awc_vol"] = (th33 - th1500).astype(np.float32)

    return ds


if __name__ == "__main__":
    pass
