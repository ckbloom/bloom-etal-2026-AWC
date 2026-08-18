"""
toth.py
=======
FC and WP computation using the pedotransfer functions of:
    Tóth, B., Weynants, M., Nemes, A., Makó, A., Bilas, G., Tóth, G. (2015).

Equations (Table 3 / Eqs. 9, 12)
---------------------------------
Field capacity regression (Eq. 9):
    theta_FC = 0.2449
               - 0.1887 * (1 / (OC + 1))
               + 0.004527 * Clay
               + 0.001535 * Silt
               + 0.001442 * Silt * (1 / (OC + 1))
               - 0.00005110 * Silt * Clay
               + 0.0008676 * Clay * (1 / (OC + 1))

Wilting point regression (Eq. 12):
    theta_WP = 0.09878
               + 0.002127 * Clay
               - 0.0008366 * Silt
               - 0.07670 * (1 / (OC + 1))
               + 0.00003853 * Silt * Clay
               + 0.002330 * Clay * (1 / (OC + 1))
               + 0.0009498 * Silt * (1 / (OC + 1))

Derived outputs:
    AWC = theta_FC - theta_WP

Key outputs
-----------
    FC  (field capacity,  ~33 kPa) = theta_FC     [Eq. 9]
    WP  (wilting point, ~1500 kPa) = theta_WP     [Eq. 12]
    AWC (available water capacity) = FC - WP

Input unit conventions
-----------------------
    Sand : sand content as PERCENTAGE (%)
    Clay : clay content as PERCENTAGE (%)
    Silt : derived internally as 100 - Sand - Clay  (%)
    OC   : organic carbon as PERCENTAGE (%)

All moisture outputs are volumetric fractions (cm³/cm³).

Calibration notes
-----------------
    PTFs calibrated on the HYPRES database (European soils).
    Validated across a broad range of European soil types and climatic
    regions. Recommended for European applications; performance in other
    regions has not been systematically evaluated.
    See Tóth et al. (2015) Table 4 for calibration/validation statistics.
"""


def apply_toth(ds):
    """
    Compute Toth et al. (2015) moisture parameters and AWC for all depth layers.

    Inputs expected in ds:
        sand  (%)  : sand content
        clay  (%)  : clay content
        soc   (%)  : soil organic carbon

    Adds to ds:
        wp      : wilting point          (cm³/cm³)
        fc      : field capacity         (cm³/cm³)
        awc_vol : plant available water  (cm³/cm³)  — fc - wp

    """

    # Calculate silt as balance
    ds['silt'] = 100 - (ds['sand'] + ds['clay'])

    # Pre-calculate common term (1 / (OC + 1))
    oc_term = 1 / (ds['soc'] + 1)

    # 1. theta_FC (Field Capacity) - Equation (9)
    # Range clip typical for volumetric water content
    ds['fc'] = (0.2449
                - 0.1887 * oc_term
                + 0.004527 * ds['clay']
                + 0.001535 * ds['silt']
                + 0.001442 * ds['silt'] * oc_term
                - 0.00005110 * ds['silt'] * ds['clay']
                + 0.0008676 * ds['clay'] * oc_term)  # .clip(0.01, 0.6)

    # 2. theta_WP (Wilting Point) - Equation (12)
    ds['wp'] = (0.09878
                + 0.002127 * ds['clay']
                - 0.0008366 * ds['silt']
                - 0.07670 * oc_term
                + 0.00003853 * ds['silt'] * ds['clay']
                + 0.002330 * ds['clay'] * oc_term
                + 0.0009498 * ds['silt'] * oc_term)  # .clip(0.0, 0.5)

    # 3. AWC (Available Water Capacity)
    # Calculated as the difference between FC and WP
    ds['awc_vol'] = (ds['fc'] - ds['wp'])  # .clip(lower=0)

    return ds


if __name__ == "__main__":
    pass
