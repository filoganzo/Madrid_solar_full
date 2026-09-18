"""
irradiance.py — from sun position to watts on a tilted panel.

Chain, in order, with the reason each step exists:

  1. Extraterrestrial irradiance          how much arrives at the top of the sky
  2. Ineichen-Perez clear-sky model       how much survives a clear atmosphere
  3. Monthly clear-sky index              how much survives Madrid's clouds
  4. Erbs decomposition                   how much is beam vs diffuse
  5. Hay-Davies transposition             how much lands on a plane at (tilt, az)
  6. ASHRAE incidence-angle modifier      how much gets through the glass

Step 3 is the only place a measured local number enters, and it is a single
scalar per month. That was deliberate: it means the model can be calibrated
against PVGIS with twelve numbers, and if the calibration needs a big
correction, the physics above it is wrong and I want to see that rather than
bury it in a fudge factor.

Result of that calibration, on the run in outputs/: a global scale of 1.00x
and annual POA within tolerance of PVGIS. See tests/test_sanity.py.
"""

from __future__ import annotations

import numpy as np

from .solar_geometry import DAYS_IN_MONTH, incidence_cosine

# Monthly global horizontal irradiance for Madrid, kWh/m2/month.
# src: PVGIS-SARAH2 monthly normals for 40.42N 3.70W, rounded. conf: high
# This is the ONLY measured local input in the whole irradiance chain. The
# model is calibrated on it and then validated against a different PVGIS
# number (plane-of-array at the optimum tilt) that it was not fitted to.
# Calibrating on one quantity and checking against another is the only way
# to tell whether the physics in between is right or whether the fit is just
# absorbing the error.
GHI_MONTHLY_TARGET = np.array([70.0, 91.0, 142.0, 173.0, 208.0, 233.0,
                               247.0, 220.0, 163.0, 109.0, 73.0, 62.0])

# Day-type mixture parameters. See daily_clearness() for why this exists.
KC_CLEAR = 0.96        # a cloudless Madrid day, as a fraction of clear-sky
KC_CLOUDY_LO = 0.15    # overcast
KC_CLOUDY_HI = 0.55    # broken cloud
DAY_TYPE_SEED = 7      # fixed, so every run is reproducible


def daily_clearness(kc_month: np.ndarray, seed: int = DAY_TYPE_SEED
                    ) -> np.ndarray:
    """A clearness index for each of the 365 days, not one per month.

    This replaces an earlier version that scaled the clear-sky curve by a
    monthly mean. That version was wrong, and wrong in an instructive way, so
    the reasoning is recorded rather than quietly deleted.

    The diffuse fraction is a convex function of the clearness index. A month
    at mean clearness 0.80 is not thirty days at 0.80; it is roughly
    twenty-four cloudless days at 0.96 and six overcast ones at 0.3. Feeding
    the monthly mean into the Erbs correlation therefore lands on the wrong
    part of the curve and overstates diffuse irradiance. In the first run it
    produced an annual diffuse fraction of 0.41 against a real Madrid value
    near 0.31, which pushed plane-of-array irradiance 7.5% below PVGIS —
    because a tilted plane earns most of its gain from the beam component,
    and the error had quietly converted beam into diffuse.

    Representing the month as a mixture of day types fixes the split and
    brings the independent PVGIS check to within half a percent. It also
    fixes a second problem for free: the battery model now sees genuine
    consecutive overcast days, which a monthly mean can never produce and
    which flatter storage badly.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(365)
    d0 = 0
    mean_cloudy = (KC_CLOUDY_LO + KC_CLOUDY_HI) / 2.0
    for m in range(12):
        n = DAYS_IN_MONTH[m]
        f = np.clip((kc_month[m] - mean_cloudy) / (KC_CLEAR - mean_cloudy),
                    0.0, 1.0)
        n_clear = int(round(f * n))
        vals = np.concatenate([
            np.full(n_clear, KC_CLEAR),
            rng.uniform(KC_CLOUDY_LO, KC_CLOUDY_HI, n - n_clear),
        ])
        # Renormalise so the monthly total still matches the target exactly.
        vals = vals * kc_month[m] / vals.mean()
        rng.shuffle(vals)
        out[d0:d0 + n] = np.clip(vals, 0.05, 1.0)
        d0 += n
    return out


def ineichen_clear_sky(sun, elevation_m: float, linke_monthly, month_idx):
    """Ineichen & Perez (2002) clear-sky GHI and DNI.

    Returns (ghi_cs, dni_cs, dhi_cs) in W/m2, zero when the sun is down.
    """
    alt = sun.altitude_deg
    zen = np.radians(sun.zenith_deg)
    am = sun.air_mass
    i0 = sun.extraterrestrial_wm2

    tl = np.asarray(linke_monthly)[month_idx]

    fh1 = np.exp(-elevation_m / 8000.0)
    fh2 = np.exp(-elevation_m / 1250.0)
    a1 = 5.09e-5 * elevation_m + 0.868
    a2 = 3.92e-5 * elevation_m + 0.0387

    cos_z = np.clip(np.cos(zen), 0.0, None)

    with np.errstate(invalid="ignore", over="ignore"):
        ghi = (a1 * i0 * cos_z
               * np.exp(-a2 * am * (fh1 + fh2 * (tl - 1.0)))
               * np.exp(0.01 * am ** 1.8))
        b = 0.664 + 0.163 / fh1
        dni = b * i0 * np.exp(-0.09 * am * (tl - 1.0))

    up = alt > 0.0
    ghi = np.where(up, np.nan_to_num(ghi), 0.0)
    dni = np.where(up, np.nan_to_num(dni), 0.0)

    # Physical consistency: the beam component cannot exceed the global.
    dni = np.minimum(dni, np.where(cos_z > 1e-6, ghi / np.maximum(cos_z, 1e-6),
                                   0.0))
    dhi = np.maximum(ghi - dni * cos_z, 0.0)
    return ghi, dni, dhi


def erbs_diffuse_fraction(kt):
    """Erbs et al. (1982): diffuse fraction of GHI as a function of the
    clearness index kt = GHI / (I0 * cos z).

    Used instead of assuming a fixed diffuse fraction because in a dense
    street the diffuse component is most of what reaches a shaded panel, so
    getting the split wrong biases shaded roofs far more than open ones.
    """
    kt = np.clip(np.nan_to_num(kt), 0.0, 1.0)
    f = np.where(
        kt <= 0.22,
        1.0 - 0.09 * kt,
        np.where(
            kt <= 0.80,
            (0.9511 - 0.1604 * kt + 4.388 * kt ** 2
             - 16.638 * kt ** 3 + 12.336 * kt ** 4),
            0.165,
        ),
    )
    return np.clip(f, 0.0, 1.0)


def monthly_clear_sky_index(ghi_clear_sky: np.ndarray, month_idx: np.ndarray,
                            ghi_target=None) -> np.ndarray:
    """The twelve numbers that carry all the local climate information."""
    if ghi_target is None:
        ghi_target = GHI_MONTHLY_TARGET
    cs = np.array([np.nansum(ghi_clear_sky[month_idx == m]) / 1000.0
                   for m in range(12)])
    return np.asarray(ghi_target) / np.where(cs > 0, cs, 1.0)


def apply_cloudiness(sun, ghi_cs, dni_cs, dhi_cs, month_idx,
                     kc_monthly=None, ghi_target=None, seed=DAY_TYPE_SEED):
    """Turn clear-sky irradiance into an all-sky hourly series.

    GHI is scaled by that day's clearness index; the beam/diffuse split is
    then re-derived from the resulting hourly clearness index rather than
    scaled proportionally. Clouds cut beam far harder than diffuse, so
    proportional scaling would overstate DNI substantially.
    """
    if kc_monthly is None:
        kc_monthly = monthly_clear_sky_index(ghi_cs, month_idx, ghi_target)
    kc_daily = daily_clearness(np.asarray(kc_monthly), seed=seed)
    kc_hourly = np.repeat(kc_daily, 24)
    if len(kc_hourly) != len(ghi_cs):
        kc_hourly = np.resize(kc_hourly, len(ghi_cs))

    ghi = ghi_cs * kc_hourly
    cos_z = np.clip(np.cos(np.radians(sun.zenith_deg)), 0.0, None)
    i0h = sun.extraterrestrial_wm2 * cos_z

    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(i0h > 1.0, ghi / i0h, 0.0)
    df = erbs_diffuse_fraction(kt)

    dhi = ghi * df
    dni = np.where(cos_z > 0.02, (ghi - dhi) / np.maximum(cos_z, 0.02), 0.0)
    return ghi, np.maximum(dni, 0.0), dhi


def iam_ashrae(cos_i, b0: float = 0.05):
    """Incidence-angle modifier. Glass reflects more at grazing angles.
    Small annually (~2%), but it is concentrated in the early and late hours,
    which in Spain is exactly where the expensive tariff periods are."""
    cos_i = np.clip(cos_i, 1e-6, 1.0)
    iam = 1.0 - b0 * (1.0 / cos_i - 1.0)
    return np.clip(iam, 0.0, 1.0)


def transpose_hay_davies(sun, ghi, dni, dhi, tilt_deg, azimuth_deg,
                         albedo: float = 0.2, sky_view: float = 1.0,
                         ground_view: float = 1.0, beam_shade=None,
                         iam_b0: float = 0.05):
    """Plane-of-array irradiance, split into its three physical parts.

    Hay & Davies rather than the isotropic model because the circumsolar
    brightening term is worth 3-5% annually on a south-facing plane, and
    because the isotropic model cannot represent the thing this study is
    actually about: a panel in a street canyon that still sees the sun but
    has lost half its view of the sky.

    sky_view / ground_view  (0-1) are the geometric view factors left after
    surrounding buildings block part of the hemisphere. Computed in
    shading.py. Applying them here, separately from the beam shading mask, is
    what stops the model from treating a shaded urban panel like a clean one
    with a flat loss factor.

    beam_shade  (0-1 array) is 1 when the sun reaches the panel and 0 when a
    building is in the way.
    """
    cos_i = incidence_cosine(sun.altitude_deg, sun.azimuth_deg,
                             tilt_deg, azimuth_deg)
    cos_z = np.clip(np.cos(np.radians(sun.zenith_deg)), 0.0, None)

    if beam_shade is None:
        beam_shade = 1.0

    iam = iam_ashrae(cos_i, iam_b0)
    poa_beam = dni * cos_i * beam_shade * iam

    # Anisotropy index: the share of the diffuse that behaves like beam.
    with np.errstate(divide="ignore", invalid="ignore"):
        ai = np.where(sun.extraterrestrial_wm2 > 0,
                      dni / sun.extraterrestrial_wm2, 0.0)
    ai = np.clip(np.nan_to_num(ai), 0.0, 1.0)
    rb = np.where(cos_z > 0.02, cos_i / np.maximum(cos_z, 0.02), 0.0)

    tilt = np.radians(tilt_deg)
    iso = (1.0 + np.cos(tilt)) / 2.0
    poa_sky = dhi * (ai * rb + (1.0 - ai) * iso * sky_view)

    poa_ground = ghi * albedo * ((1.0 - np.cos(tilt)) / 2.0) * ground_view

    poa = poa_beam + poa_sky + poa_ground
    return {
        "poa": poa,
        "beam": poa_beam,
        "sky_diffuse": poa_sky,
        "ground": poa_ground,
        "cos_incidence": cos_i,
        "iam": iam,
    }
