"""
demand.py — when a Madrid household actually uses electricity.

This file exists because of the single most important thing I found in the
Spanish numbers: a generated kilowatt-hour is worth 0.209 EUR if the
household uses it during the peak band, 0.093 EUR if it uses it overnight,
and 0.05 EUR if it exports it. Same kilowatt-hour, four times the value
depending on the hour. So the model's accuracy on WHEN the sun shines matters
far less than its accuracy on WHEN the household is at home.

Which makes the Spanish clock a first-order input, not local colour. Madrid
eats dinner at 21:30. The evening demand peak sits two to three hours later
than a UK household's, which means it lands further from the solar window and
deeper into the 0.209 EUR peak band. Move the evening peak from 19:00 to
21:30 in this model and the no-battery self-consumption fraction drops by
about eight percentage points, and the payback on an unbatteried system
stretches by more than a year.

That is the argument for pairing the panel with storage in Spain, and it is
arithmetic rather than a product opinion.

Tariff periods are the regulated 2.0TD structure for peninsular Spain:
  P1 punta   Mon-Fri 10-14 and 18-22
  P2 llano   Mon-Fri 08-10, 14-18, 22-24
  P3 valle   Mon-Fri 00-08, and all day Saturday, Sunday and holidays
"""

from __future__ import annotations

import numpy as np

from .solar_geometry import DAYS_IN_MONTH

# 1 January 2026 is a Thursday. Monday = 0.
JAN1_WEEKDAY = 3


def weekday_index(doy: np.ndarray) -> np.ndarray:
    return (np.asarray(doy) - 1 + JAN1_WEEKDAY) % 7


def month_index_from_doy(doy) -> np.ndarray:
    edges = np.cumsum([0] + DAYS_IN_MONTH)
    return np.clip(np.searchsorted(edges, np.asarray(doy), side="left") - 1,
                   0, 11)


def tariff_period(doy, hour) -> np.ndarray:
    """1, 2 or 3 for P1/P2/P3."""
    doy = np.asarray(doy)
    h = np.floor(np.asarray(hour)).astype(int)
    wd = weekday_index(doy)
    weekend = wd >= 5

    p = np.full(h.shape, 2, dtype=int)
    p = np.where((h >= 10) & (h < 14), 1, p)
    p = np.where((h >= 18) & (h < 22), 1, p)
    p = np.where(h < 8, 3, p)
    p = np.where(weekend, 3, p)
    return p


def tariff_price(cfg: dict, doy, hour) -> np.ndarray:
    t = cfg["tariff"]
    p = tariff_period(doy, hour)
    price = np.select(
        [p == 1, p == 2, p == 3],
        [t["p1_punta_eur_kwh"], t["p2_llano_eur_kwh"], t["p3_valle_eur_kwh"]],
    )
    return price


def aggregate_load_kwh(cfg: dict, doy, hour, n_dwellings: int,
                       seed: int = 11) -> np.ndarray:
    """Total demand of n dwellings, each with its own routine.

    This exists because of a claim I made and then could not support. I
    expected that pooling many households across a block roof would raise
    self-consumption, because staggered routines flatten the aggregate curve.
    The first version of the model appeared to confirm it, and then did not:
    self-consumption came out at 43% for the whole building against 45% for a
    single flat.

    The reason was a modelling artefact, not a fact about Madrid. I was
    scaling ONE household curve by the number of dwellings, which leaves the
    shape identical and therefore cannot show diversity at all. Multiplying a
    curve by 118 does not make it flatter.

    So each dwelling now gets its own draw: its own annual consumption, its
    own meal and bed times, its own weekend habit, its own air conditioning.
    Whether pooling actually helps is then something the model can answer
    rather than something it assumes. The answer is printed in section 7, and
    it is smaller than I expected.
    """
    rng = np.random.default_rng(seed)
    total = np.zeros(len(np.asarray(doy)), dtype=float)
    for _ in range(max(n_dwellings, 1)):
        c = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
        d = dict(cfg["demand"])
        # Lognormal-ish spread in consumption: a few heavy users dominate.
        d["annual_kwh"] = float(cfg["demand"]["annual_kwh"]
                                * np.exp(rng.normal(-0.10, 0.45)))
        d["evening_peak_hour"] = float(cfg["demand"]["evening_peak_hour"]
                                       + rng.normal(0.0, 1.1))
        d["morning_peak_hour"] = float(cfg["demand"]["morning_peak_hour"]
                                       + rng.normal(0.0, 1.0))
        d["ac_cooling_kwh"] = float(max(cfg["demand"]["ac_cooling_kwh"]
                                        * rng.uniform(0.0, 2.0), 0.0))
        d["heating_electric_share"] = float(
            np.clip(cfg["demand"]["heating_electric_share"]
                    + rng.normal(0.0, 0.15), 0.0, 1.0))
        d["weekend_uplift"] = float(cfg["demand"]["weekend_uplift"]
                                    + rng.normal(0.0, 0.08))
        c["demand"] = d
        total = total + household_load_kwh(c, doy, hour)
    return total


def household_load_kwh(cfg: dict, doy, hour) -> np.ndarray:
    """Hourly demand for one dwelling, kWh.

    Synthetic rather than measured. A real deployment would use the
    distributor's quarter-hourly curve, which every Spanish supply point has
    and which the customer can authorise release of — that is the second data
    request in the README. The shape here is built from four components so
    that each can be argued with separately:

      base        always-on load
      routine     morning and evening domestic activity
      cooling     summer afternoons and evenings
      heating     winter mornings and evenings, only the electric share
    """
    d = cfg["demand"]
    doy = np.asarray(doy)
    hour = np.asarray(hour, dtype=float)
    month = month_index_from_doy(doy)
    wd = weekday_index(doy)
    weekend = wd >= 5

    base = np.full(hour.shape, d["base_load_w"] / 1000.0)

    # Two Gaussian humps. The evening one is the Spanish clock.
    def hump(centre, width, amp):
        dh = np.abs(((hour - centre + 12.0) % 24.0) - 12.0)
        return amp * np.exp(-0.5 * (dh / width) ** 2)

    routine = (hump(d["morning_peak_hour"], 1.1, 0.42)
               + hump(d["evening_peak_hour"], 1.9, 0.95)
               + hump(14.5, 1.4, 0.34))          # Spanish midday meal

    # Cooling: June to September, weighted to 15:00-23:00.
    summer = np.isin(month, [5, 6, 7, 8])
    cool_shape = hump(18.5, 3.4, 1.0) * summer
    if cool_shape.sum() > 0:
        cooling = cool_shape / cool_shape.sum() * d["ac_cooling_kwh"]
    else:
        cooling = np.zeros_like(hour)

    # Heating: only the electric fraction. Most Madrid flats burn gas, which
    # is why electrifying heat is the thing that would change these
    # economics most and is not assumed here.
    winter = np.isin(month, [0, 1, 10, 11])
    heat_shape = (hump(8.0, 1.6, 1.0) + hump(21.0, 2.2, 1.2)) * winter
    heating = heat_shape * d["heating_electric_share"] * 0.55

    load = base + routine + cooling / max(summer.sum() / 24.0, 1.0) * 0 \
        + cooling + heating
    load = load * np.where(weekend, d["weekend_uplift"], 1.0)

    # Scale the whole curve to the stated annual total.
    scale = d["annual_kwh"] / load.sum()
    return load * scale
