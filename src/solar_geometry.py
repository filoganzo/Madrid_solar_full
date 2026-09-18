"""
solar_geometry.py — where the sun is, from first principles.

No pvlib, no lookup tables. The point of writing this by hand is that every
downstream number depends on it, so I wanted to be able to check it against
something I can verify without trusting a package: at the equinox, solar noon
altitude must equal 90 - latitude, exactly. tests/test_sanity.py asserts it.

Implements the NOAA / Meeus low-precision solar position algorithm. Accurate
to roughly 0.01 deg over 1950-2050, which is two orders of magnitude better
than anything else in this model needs.

Conventions used throughout the repo:
    azimuth  0 deg = due south, negative = east of south, positive = west.
             (This matches PVGIS, which is what we calibrate against. It is
             NOT the compass convention. Getting this wrong flips the whole
             city east-west and the error is invisible in annual totals —
             which is exactly why it is worth a comment.)
    altitude degrees above the horizon, negative below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def day_of_year(month: int, day: int) -> int:
    """1-365. Non-leap year: a leap day changes annual yield by 0.27%, which
    is well inside every other uncertainty here, so it is ignored on purpose."""
    return sum(DAYS_IN_MONTH[: month - 1]) + day


def month_day(doy: int) -> tuple[int, int]:
    m = 0
    d = doy
    while d > DAYS_IN_MONTH[m]:
        d -= DAYS_IN_MONTH[m]
        m += 1
    return m + 1, d


def is_summer_time(doy: int) -> bool:
    """EU daylight saving: last Sunday of March to last Sunday of October.
    Approximated by fixed day-of-year boundaries (86 / 303) for a generic
    year. Matters because the tariff periods below are in local clock time,
    and an hour of error moves a kWh between the 0.209 and 0.118 bands."""
    return 86 <= doy <= 303


def julian_day(year: int, doy: int, hour_utc: float) -> float:
    month, day = month_day(doy)
    y, m = year, month
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    jd = (math.floor(365.25 * (y + 4716))
          + math.floor(30.6001 * (m + 1))
          + day + b - 1524.5)
    return jd + hour_utc / 24.0


# --------------------------------------------------------------------------
# Sun position
# --------------------------------------------------------------------------

@dataclass
class SunPosition:
    altitude_deg: np.ndarray
    azimuth_deg: np.ndarray      # 0 = south, +W, -E
    zenith_deg: np.ndarray
    declination_deg: np.ndarray
    eq_time_min: np.ndarray
    air_mass: np.ndarray
    extraterrestrial_wm2: np.ndarray


def sun_position(lat: float, lon: float, year: int, doy, hour_local,
                 utc_offset: float | np.ndarray) -> SunPosition:
    """Vectorised. doy, hour_local and utc_offset broadcast together."""
    doy = np.asarray(doy, dtype=float)
    hour_local = np.asarray(hour_local, dtype=float)
    utc_offset = np.asarray(utc_offset, dtype=float)

    hour_utc = hour_local - utc_offset

    # Julian century since J2000.0
    jd = np.vectorize(julian_day)(year, doy.astype(int), 0.0) + hour_utc / 24.0
    t = (jd - 2451545.0) / 36525.0

    # Geometric mean longitude and anomaly of the Sun (deg)
    L0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    M = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    Mr = np.radians(M)

    # Equation of the centre -> true longitude
    C = (np.sin(Mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
         + np.sin(2 * Mr) * (0.019993 - 0.000101 * t)
         + np.sin(3 * Mr) * 0.000289)
    true_long = L0 + C

    # Apparent longitude (nutation + aberration)
    omega = 125.04 - 1934.136 * t
    app_long = true_long - 0.00569 - 0.00478 * np.sin(np.radians(omega))

    # Obliquity of the ecliptic
    seconds = 21.448 - t * (46.8150 + t * (0.00059 - t * 0.001813))
    eps0 = 23.0 + (26.0 + seconds / 60.0) / 60.0
    eps = eps0 + 0.00256 * np.cos(np.radians(omega))

    # Declination
    decl = np.degrees(np.arcsin(np.sin(np.radians(eps))
                                * np.sin(np.radians(app_long))))

    # Equation of time (minutes)
    y_term = np.tan(np.radians(eps / 2.0)) ** 2
    ecc = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    eq_time = 4.0 * np.degrees(
        y_term * np.sin(2 * np.radians(L0))
        - 2 * ecc * np.sin(Mr)
        + 4 * ecc * y_term * np.sin(Mr) * np.cos(2 * np.radians(L0))
        - 0.5 * y_term ** 2 * np.sin(4 * np.radians(L0))
        - 1.25 * ecc ** 2 * np.sin(2 * Mr)
    )

    # True solar time -> hour angle.
    # lon is degrees EAST positive; Madrid is negative, i.e. Madrid sits ~15
    # deg west of the centre of its own time zone, so solar noon falls near
    # 13:50 local clock time in summer. This is not a rounding detail: it is
    # why a "south-facing" Spanish roof produces its peak in the afternoon
    # tariff band rather than the midday one.
    true_solar_time = (hour_utc * 60.0 + eq_time + 4.0 * lon) % 1440.0
    hour_angle = true_solar_time / 4.0 - 180.0
    ha = np.radians(hour_angle)

    latr = np.radians(lat)
    dr = np.radians(decl)

    cos_zen = np.sin(latr) * np.sin(dr) + np.cos(latr) * np.cos(dr) * np.cos(ha)
    cos_zen = np.clip(cos_zen, -1.0, 1.0)
    zenith = np.degrees(np.arccos(cos_zen))
    altitude = 90.0 - zenith

    # Atmospheric refraction near the horizon. Small, but it decides whether
    # the first and last half hour of the day exist at all.
    alt_r = np.radians(np.maximum(altitude, -1.0))
    refract = np.where(
        altitude > 85.0, 0.0,
        np.where(altitude > 5.0,
                 (58.1 / np.tan(alt_r) - 0.07 / np.tan(alt_r) ** 3
                  + 0.000086 / np.tan(alt_r) ** 5) / 3600.0,
                 np.where(altitude > -0.575,
                          (1735.0 + altitude * (-518.2 + altitude
                           * (103.4 + altitude * (-12.79 + altitude * 0.711))))
                          / 3600.0,
                          -20.772 / np.tan(alt_r) / 3600.0))
    )
    altitude = altitude + np.where(altitude > -1.0, refract, 0.0)

    # Azimuth, in the south-zero convention.
    denom = np.cos(latr) * np.sin(np.radians(zenith))
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    cos_az = (np.sin(dr) - np.sin(latr) * cos_zen) / denom
    cos_az = np.clip(cos_az, -1.0, 1.0)
    az_from_north = np.degrees(np.arccos(cos_az))
    az_from_north = np.where(hour_angle > 0.0, 360.0 - az_from_north,
                             az_from_north)
    # az_from_north is now the compass bearing, 0 = north, clockwise.
    # Converting to the south-zero convention is a subtraction, not a
    # negation. The first version wrote `180 - az_from_north`, which flipped
    # east and west. Annual energy totals barely moved, because the array
    # azimuth is -6 deg and the sun's path is nearly symmetric about noon —
    # so the error was invisible in every headline number and only showed up
    # in test_azimuth_convention_is_south_zero, which checks the sign at
    # 08:00 directly. It mattered: with east and west swapped, every
    # shading calculation put the morning obstruction on the wrong side of
    # the street.
    azimuth = az_from_north - 180.0          # -> 0 = south, +W, -E
    azimuth = np.where(azimuth > 180.0, azimuth - 360.0, azimuth)
    azimuth = np.where(azimuth < -180.0, azimuth + 360.0, azimuth)

    # Kasten-Young relative air mass, pressure-corrected for altitude.
    alt_pos = np.maximum(altitude, 0.5)
    am = 1.0 / (np.sin(np.radians(alt_pos))
                + 0.50572 * (alt_pos + 6.07995) ** -1.6364)
    am = np.where(altitude > 0.0, am, np.nan)

    # Extraterrestrial normal irradiance, with the eccentricity correction.
    e0 = 1.00011 + 0.034221 * np.cos(2 * np.pi * doy / 365.0) \
        + 0.00128 * np.sin(2 * np.pi * doy / 365.0) \
        + 0.000719 * np.cos(4 * np.pi * doy / 365.0) \
        + 0.000077 * np.sin(4 * np.pi * doy / 365.0)
    dni_et = 1361.0 * e0

    return SunPosition(altitude, azimuth, zenith, decl, eq_time, am, dni_et)


def sunrise_sunset_local(lat: float, lon: float, year: int, doy: int,
                         utc_offset: float) -> tuple[float, float]:
    """Local clock hours of sunrise and sunset. Found by bisection on the
    altitude function rather than by a closed form, because the closed form
    ignores refraction and I want the two to agree."""
    hrs = np.linspace(0.0, 24.0, 24 * 60 + 1)
    alt = sun_position(lat, lon, year, np.full_like(hrs, doy), hrs,
                       utc_offset).altitude_deg
    above = alt > -0.833                       # standard sun-disc refraction
    if not above.any():
        return float("nan"), float("nan")
    idx = np.where(above)[0]
    return float(hrs[idx[0]]), float(hrs[idx[-1]])


def incidence_cosine(sun_alt_deg, sun_az_deg, tilt_deg, surface_az_deg):
    """cos of the angle between the sun and a tilted plane's normal.
    Clipped at zero: a panel does not produce from behind."""
    alt = np.radians(sun_alt_deg)
    az = np.radians(sun_az_deg)
    tilt = np.radians(tilt_deg)
    saz = np.radians(surface_az_deg)
    cos_i = (np.sin(alt) * np.cos(tilt)
             + np.cos(alt) * np.sin(tilt) * np.cos(az - saz))
    return np.clip(cos_i, 0.0, None)
