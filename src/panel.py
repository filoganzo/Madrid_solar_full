"""
panel.py — irradiance in, kilowatt-hours out, for one module.

The unit of analysis here is deliberately one panel, not one system. Two
reasons.

First, physical: in a dense block the panels on one roof do not have the same
yield. The row behind the stairhead can produce 25% less than the row on the
south edge. An average over the roof hides the fact that the last row added
is often the one that loses money.

Second, commercial: the decision Fuse would actually face is "how many panels
do we put on this roof", and that is a marginal question. Marginal revenue
per panel has to meet marginal cost per panel. You cannot see that in a
system-level model, and it is the reason economics.py costs panels marginally
rather than at an average EUR/Wp.

Madrid-specific choice worth flagging: wind speed at a sheltered urban
rooftop is set to 1.3 m/s, not the 3 m/s an open-field model would use.
Cells run hotter in a courtyard than in a field, and at 26 C July ambient
that is worth about 2% of summer output. Small, but it is the kind of thing
that makes a model built for a field wrong in a city.
"""

from __future__ import annotations

import numpy as np

# Madrid-Retiro monthly mean air temperature, deg C.
# src: AEMET climate normals, rounded.  conf: high
T_AMBIENT_MONTHLY = np.array([6.2, 7.9, 11.2, 13.2, 17.5, 23.3,
                              26.8, 26.3, 21.9, 16.0, 10.2, 6.9])

# Half the mean diurnal range, deg C. Madrid is continental: the swing is
# large, and it peaks in summer, which is when it costs the most yield.
T_DIURNAL_AMP = np.array([4.5, 5.2, 6.0, 6.3, 6.8, 7.4,
                          7.8, 7.6, 7.0, 6.0, 5.0, 4.4])

URBAN_ROOFTOP_WIND_MS = 1.3


def ambient_temperature(month_idx, hour_local):
    """Sinusoidal diurnal cycle around the monthly mean, minimum at 06:00,
    maximum at 16:00. Crude, and good enough: the model's sensitivity to a
    1 C temperature error is 0.29% of output."""
    month_idx = np.asarray(month_idx)
    hour = np.asarray(hour_local, dtype=float)
    mean = T_AMBIENT_MONTHLY[month_idx]
    amp = T_DIURNAL_AMP[month_idx]
    phase = 2.0 * np.pi * (hour - 16.0) / 24.0
    return mean + amp * np.cos(phase)


def cell_temperature(poa_wm2, t_air_c, u0: float = 25.0, u1: float = 6.84,
                     wind_ms: float = URBAN_ROOFTOP_WIND_MS):
    """Faiman (2008) module temperature.

    Chosen over the simpler NOCT method because it separates radiative and
    convective cooling, so the urban wind-speed assumption above actually
    enters the arithmetic rather than being buried in a single coefficient.
    """
    return t_air_c + poa_wm2 / (u0 + u1 * wind_ms)


def dc_power_w(poa_wm2, t_cell_c, cfg: dict):
    """DC output of one module, watts."""
    p = cfg["panel"]
    eff_ratio = 1.0 + p["temp_coeff_pmp"] * (np.asarray(t_cell_c) - 25.0)
    return np.asarray(poa_wm2) * p["area_m2"] * p["efficiency_stc"] \
        * np.clip(eff_ratio, 0.0, None)


def system_loss_factor(cfg: dict) -> float:
    """Multiplicative chain of the losses that are genuinely constant.

    Kept multiplicative rather than additive because that is what the physics
    does, and because the difference between 1-(a+b+c) and (1-a)(1-b)(1-c) on
    this set of numbers is 0.3% — small, but free to get right.
    """
    l = cfg["losses"]
    f = 1.0
    for k in ("soiling", "inverter", "dc_wiring", "ac_wiring", "mismatch",
              "availability"):
        f *= (1.0 - l[k])
    return f


def ac_energy_kwh(poa_wm2, month_idx, hour_local, cfg: dict,
                  hours_per_step: float = 1.0):
    """AC kilowatt-hours per module per time step.

    Note what is NOT here: no inverter clipping. At 450 Wp per module against
    a string inverter sized at roughly 1.15 DC/AC, clipping in Madrid costs
    under 0.5% annually and only on clear June days. It is omitted and
    stated, rather than modelled badly.
    """
    t_air = ambient_temperature(month_idx, hour_local)
    t_cell = cell_temperature(poa_wm2, t_air, cfg["panel"]["faiman_u0"],
                              cfg["panel"]["faiman_u1"])
    dc = dc_power_w(poa_wm2, t_cell, cfg)
    ac = dc * system_loss_factor(cfg)
    return ac * hours_per_step / 1000.0, t_cell


def degradation_curve(cfg: dict) -> np.ndarray:
    """Output multiplier for each year of life.

    Two-stage because that is how modules actually behave: a first-year
    light-induced drop, then a slow linear decline. A single 0.5%/yr line
    overstates lifetime yield by about 1.5%.
    """
    p = cfg["panel"]
    years = np.arange(1, p["life_years"] + 1)
    factor = (1.0 - p["degradation_yr1"]) \
        * (1.0 - p["degradation_annual"]) ** (years - 1)
    return factor
