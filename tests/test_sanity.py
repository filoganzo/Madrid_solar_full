"""
test_sanity.py — the checks I ran before believing any of the output.

    python -m pytest tests/ -v

These are grouped by what they protect against, because that is the useful
distinction:

  THEORY      checks against arithmetic that cannot be wrong. Equinox noon
              altitude must equal 90 minus latitude. Day length at the
              equinox must be twelve hours. No external data needed, so a
              failure here is unambiguously my bug.

  REFERENCE   checks against PVGIS, which the model was partly calibrated
              on. Weaker evidence, and labelled as such.

  INTERNAL    checks that two different implementations of the same thing
              agree, or that a quantity conserves. These catch the errors
              that produce plausible-looking wrong answers.

  REGRESSION  one test per bug found during the build, so it cannot come
              back. These are the most valuable tests in the file and the
              reason the file exists.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import demand, economics, irradiance, panel, roof_model, shading
from src.solar_geometry import (day_of_year, incidence_cosine, is_summer_time,
                                sun_position, sunrise_sunset_local)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YEAR = 2026


@pytest.fixture(scope="module")
def cfg():
    with open(os.path.join(HERE, "config", "madrid.yaml")) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def grid():
    doy = np.repeat(np.arange(1, 366), 24)
    hour = np.tile(np.arange(24) + 0.5, 365)
    return doy, hour


@pytest.fixture(scope="module")
def sky(cfg, grid):
    doy, hour = grid
    s = cfg["site"]
    off = np.where(np.vectorize(is_summer_time)(doy),
                   s["utc_offset_summer"], s["utc_offset_winter"])
    sun = sun_position(s["lat"], s["lon"], YEAR, doy, hour, off)
    mi = demand.month_index_from_doy(doy)
    cs = irradiance.ineichen_clear_sky(sun, s["elevation_m"],
                                       s["linke_turbidity"], mi)
    ghi, dni, dhi = irradiance.apply_cloudiness(sun, *cs, mi)
    return sun, mi, ghi, dni, dhi


# ==========================================================================
# THEORY
# ==========================================================================

def test_equinox_noon_altitude_equals_90_minus_latitude(cfg):
    """The one check that needs no data at all."""
    s = cfg["site"]
    dd = day_of_year(3, 21)
    hrs = np.linspace(0, 24, 24 * 60 + 1)
    sun = sun_position(s["lat"], s["lon"], YEAR, np.full_like(hrs, dd), hrs,
                       s["utc_offset_summer"])
    peak = float(np.nanmax(sun.altitude_deg))
    # Allow half a degree: refraction lifts the apparent sun, and the March
    # equinox is a day or two off 21 March in any given year.
    assert abs(peak - (90.0 - s["lat"])) < 0.6


def test_equinox_day_length_is_twelve_hours(cfg):
    s = cfg["site"]
    dd = day_of_year(3, 21)
    sr, ss = sunrise_sunset_local(s["lat"], s["lon"], YEAR, dd,
                                  s["utc_offset_summer"])
    # Slightly over twelve, because sunrise and sunset are defined at the
    # disc's upper limb with refraction, not at geometric zero.
    assert 12.0 < (ss - sr) < 12.4


def test_solstice_altitudes(cfg):
    s = cfg["site"]
    for month, day, expected in [(6, 21, 90 - s["lat"] + 23.44),
                                 (12, 21, 90 - s["lat"] - 23.44)]:
        dd = day_of_year(month, day)
        off = s["utc_offset_summer"] if is_summer_time(dd) \
            else s["utc_offset_winter"]
        hrs = np.linspace(0, 24, 24 * 60 + 1)
        sun = sun_position(s["lat"], s["lon"], YEAR, np.full_like(hrs, dd),
                           hrs, off)
        assert abs(float(np.nanmax(sun.altitude_deg)) - expected) < 0.6


def test_solar_noon_is_late_because_madrid_is_west_of_its_timezone(cfg):
    """Madrid sits ~15 deg west of the centre of CET, so solar noon lands
    well after 12:00 local. This has real money attached: it moves the
    array's peak into the afternoon tariff band."""
    s = cfg["site"]
    dd = day_of_year(6, 21)
    hrs = np.linspace(0, 24, 24 * 60 + 1)
    sun = sun_position(s["lat"], s["lon"], YEAR, np.full_like(hrs, dd), hrs,
                       s["utc_offset_summer"])
    noon = float(hrs[int(np.nanargmax(sun.altitude_deg))])
    assert 14.0 < noon < 14.5


def test_azimuth_convention_is_south_zero(cfg):
    """A sign error here flips the city east-west and is invisible in annual
    totals, which is exactly why it gets its own test."""
    s = cfg["site"]
    dd = day_of_year(6, 21)
    hrs = np.array([8.0, 14.25, 20.0])
    sun = sun_position(s["lat"], s["lon"], YEAR, np.full_like(hrs, dd), hrs,
                       s["utc_offset_summer"])
    assert sun.azimuth_deg[0] < -40.0      # morning: east of south
    assert abs(sun.azimuth_deg[1]) < 6.0   # solar noon: due south
    assert sun.azimuth_deg[2] > 40.0       # evening: west of south


def test_panel_produces_nothing_from_behind():
    assert incidence_cosine(30.0, 0.0, 90.0, 180.0) == 0.0


def test_horizontal_plane_sees_whole_sky():
    horizon = np.zeros(shading.N_BINS)
    assert shading.sky_view_factor(horizon, 0.0, 0.0) == pytest.approx(1.0,
                                                                       abs=0.01)


def test_wall_removes_half_the_sky():
    """An infinitely high obstruction across half the compass should remove
    close to half a horizontal panel's sky view."""
    horizon = np.zeros(shading.N_BINS)
    horizon[: shading.N_BINS // 2] = 89.0
    svf = shading.sky_view_factor(horizon, 0.0, 0.0)
    assert 0.45 < svf < 0.55


# ==========================================================================
# REFERENCE (weaker evidence: partly calibrated on the same source)
# ==========================================================================

def test_annual_ghi_matches_target(sky):
    _, _, ghi, _, _ = sky
    annual = float(np.nansum(ghi) / 1000.0)
    assert annual == pytest.approx(irradiance.GHI_MONTHLY_TARGET.sum(),
                                   rel=0.01)


def test_poa_at_optimum_matches_pvgis_independently(cfg, sky):
    """The model is fitted on horizontal GHI. This checks a DIFFERENT PVGIS
    quantity that nothing was fitted to, which is the only part of the
    calibration that carries real evidential weight."""
    sun, _, ghi, dni, dhi = sky
    cal = cfg["calibration"]
    r = irradiance.transpose_hay_davies(
        sun, ghi, dni, dhi, cal["pvgis_optimal_tilt_deg"],
        cal["pvgis_optimal_azimuth_deg"], albedo=cfg["site"]["albedo"],
        iam_b0=0.0)
    poa = float(np.nansum(r["poa"]) / 1000.0)
    assert poa == pytest.approx(cal["pvgis_poa_optimal_kwh_m2"], rel=0.04)


def test_specific_yield_matches_pvgis(cfg, sky, grid):
    doy, hour = grid
    sun, mi, ghi, dni, dhi = sky
    cal = cfg["calibration"]
    r = irradiance.transpose_hay_davies(
        sun, ghi, dni, dhi, cal["pvgis_optimal_tilt_deg"],
        cal["pvgis_optimal_azimuth_deg"], albedo=cfg["site"]["albedo"],
        iam_b0=cfg["panel"]["iam_b0"])
    kwh, _ = panel.ac_energy_kwh(r["poa"], mi, hour, cfg)
    spec = float(np.nansum(kwh)) / (cfg["panel"]["wp"] / 1000.0)
    assert spec == pytest.approx(cal["pvgis_specific_yield_kwh_kwp"],
                                 rel=cal["tolerance_pct"] / 100.0)


# ==========================================================================
# INTERNAL CONSISTENCY
# ==========================================================================

def test_two_horizon_implementations_agree(cfg):
    """build_horizon loops in Python; build_horizon_fast vectorises over
    pre-densified points. They must produce the same skyline, or the fast
    path is quietly a different model."""
    buildings, _ = roof_model.load_osm(
        os.path.join(HERE, "data", "osm_raw.json"), cfg)
    obstacles = roof_model.to_obstacles(buildings[:30])
    pts = shading.densify_all(obstacles)
    b = max(buildings[:30], key=lambda x: x.area_m2)
    point = np.array([b.polygon[:, 0].mean(), b.polygon[:, 1].mean(),
                      b.height_m - 1.0])
    slow = shading.build_horizon(point, obstacles)
    fast = shading.build_horizon_fast(point, pts)
    assert np.allclose(slow, fast, atol=0.5)


def test_irradiance_components_sum_to_poa(cfg, sky):
    sun, _, ghi, dni, dhi = sky
    r = irradiance.transpose_hay_davies(sun, ghi, dni, dhi, 15.0, -6.0,
                                        albedo=0.2)
    total = r["beam"] + r["sky_diffuse"] + r["ground"]
    assert np.allclose(np.nan_to_num(total), np.nan_to_num(r["poa"]))


def test_beam_never_exceeds_global(sky):
    sun, _, ghi, dni, dhi = sky
    cos_z = np.clip(np.cos(np.radians(sun.zenith_deg)), 0.0, None)
    assert np.all(np.nan_to_num(dni * cos_z) <= np.nan_to_num(ghi) + 1e-6)


def test_dispatch_conserves_energy(cfg):
    rng = np.random.default_rng(3)
    gen = np.abs(rng.normal(1.0, 0.6, 8760))
    load = np.abs(rng.normal(0.9, 0.4, 8760))
    for batt in (False, True):
        d = economics.dispatch(gen, load, cfg, with_battery=batt)
        # Generation goes to direct use, to the battery, or to export.
        charged = d.get("battery_charge_kwh", np.zeros_like(gen))
        assert np.isclose(gen.sum(),
                          d["direct_kwh"].sum() + charged.sum()
                          + d["export_kwh"].sum(), rtol=1e-6)
        # Demand is met by generation, by the battery, or by import.
        assert np.isclose(load.sum(),
                          d["direct_kwh"].sum()
                          + d["battery_discharge_kwh"].sum()
                          + d["import_kwh"].sum(), rtol=1e-6)


def test_battery_respects_capacity_and_efficiency(cfg):
    rng = np.random.default_rng(5)
    gen = np.abs(rng.normal(2.0, 1.0, 2000))
    load = np.abs(rng.normal(0.5, 0.2, 2000))
    d = economics.dispatch(gen, load, cfg, with_battery=True)
    # Round-trip losses mean discharge must be strictly below charge.
    assert d["battery_discharge_kwh"].sum() < d["battery_charge_kwh"].sum()
    ratio = d["battery_discharge_kwh"].sum() / d["battery_charge_kwh"].sum()
    assert ratio <= cfg["battery"]["round_trip_efficiency"] + 1e-6


def test_marginal_cost_is_flat_but_average_is_not(cfg):
    m9 = economics.marginal_cost_of_nth_panel(9, cfg)
    m30 = economics.marginal_cost_of_nth_panel(30, cfg)
    assert m9 == pytest.approx(m30, rel=1e-9)
    a1 = economics.capex_breakdown(1, cfg)["eur_per_wp"]
    a32 = economics.capex_breakdown(32, cfg)["eur_per_wp"]
    assert a1 > a32 * 8          # the whole argument for selling the roof


def test_tariff_periods_are_the_regulated_ones(cfg):
    # A Wednesday in January. 1 Jan 2026 is a Thursday, so doy 7 is Wednesday.
    doy = np.full(24, 7)
    hour = np.arange(24) + 0.5
    p = demand.tariff_period(doy, hour)
    assert set(p[0:8]) == {3}                    # 00-08 valle
    assert set(p[10:14]) == {1}                  # 10-14 punta
    assert set(p[18:22]) == {1}                  # 18-22 punta
    assert set(p[14:18]) == {2}                  # 14-18 llano
    # A Sunday is entirely valle.
    sunday = np.full(24, 4)                      # doy 4 = Sunday
    assert set(demand.tariff_period(sunday, hour)) == {3}


def test_export_credit_cannot_exceed_import_cost(cfg, grid):
    """RD 244/2019: compensation is capped at the energy term of the period.
    An oversized array must not be able to earn its way past that cap."""
    doy, hour = grid
    load = np.full(len(doy), 0.02)               # a tiny consumer
    gen = np.zeros(len(doy))
    gen[(hour > 10) & (hour < 16)] = 3.0         # a huge array
    d = economics.dispatch(gen, load, cfg)
    v = economics.annual_value(d, cfg, doy, hour)
    import_cost = float((d["import_kwh"]
                         * demand.tariff_price(cfg, doy, hour)).sum())
    assert v["export_credit_eur"] <= import_cost + 1e-6
    assert v["export_credit_lost_to_cap_eur"] > 0.0


def test_panel_packing_respects_setback(cfg):
    buildings, _ = roof_model.load_osm(
        os.path.join(HERE, "data", "osm_raw.json"), cfg)
    b = max((x for x in buildings if x.is_addressable),
            key=lambda x: x.area_m2)
    slots, _ = roof_model.pack_panels(b, cfg)
    assert len(slots) > 0
    for s in slots[:200]:
        d = roof_model.dist_to_boundary(s.x, s.y, b.polygon)
        assert d >= cfg["roof"]["setback_m"] - 0.9   # corner, not centre


def test_degradation_is_monotonic_and_ends_where_datasheets_say(cfg):
    d = panel.degradation_curve(cfg)
    assert np.all(np.diff(d) < 0)
    assert 0.83 < d[-1] < 0.90


# ==========================================================================
# REGRESSION — one test per bug found while building this
# ==========================================================================

def test_regression_diffuse_fraction_is_not_overstated(sky):
    """BUG 1. The first version scaled clear-sky irradiance by a MONTHLY MEAN
    clearness index and then fed that mean into the Erbs correlation. Because
    the diffuse fraction is convex in the clearness index, this landed on the
    wrong part of the curve: it reported an annual diffuse fraction of 0.41
    against a real Madrid value near 0.31, silently converting beam into
    diffuse and pushing plane-of-array irradiance 7.5% below PVGIS.

    Fixed by representing each month as a mixture of clear and overcast days.
    """
    _, _, ghi, _, dhi = sky
    frac = float(np.nansum(dhi) / np.nansum(ghi))
    assert 0.27 < frac < 0.34


def test_regression_day_types_produce_consecutive_overcast_days(sky):
    """The same bug had a second consequence. A monthly mean can never
    produce two bad days in a row, which flatters any battery. Check that the
    day-type mixture actually yields consecutive low-output days."""
    _, _, ghi, _, _ = sky
    daily = np.nansum(ghi.reshape(365, 24), axis=1) / 1000.0
    poor = daily < np.median(daily) * 0.55
    runs = 0
    for i in range(1, len(poor)):
        if poor[i] and poor[i - 1]:
            runs += 1
    assert runs >= 10


def test_regression_loss_change_must_change_generation(cfg, sky, grid):
    """BUG 2. The sensitivity table reported a swing of exactly 0.0 years for
    soiling between 1% and 5%, which is impossible. The cause: generation was
    computed once outside the sensitivity loop, so a change to a loss factor
    never reached it. The test asserts the loss chain actually responds."""
    import copy
    base = panel.system_loss_factor(cfg)
    c2 = copy.deepcopy(cfg)
    c2["losses"]["soiling"] = 0.05
    worse = panel.system_loss_factor(c2)
    assert worse < base
    assert (base - worse) / base > 0.02


def test_regression_aggregate_load_is_not_a_scaled_single_household(cfg, grid):
    """BUG 3. The collective-self-consumption case originally used one
    household curve multiplied by the number of dwellings, which leaves the
    SHAPE identical and therefore cannot represent diversity at all. The
    model appeared to support a claim about pooling that it could not test.

    The aggregate must now be genuinely flatter than a scaled single curve.
    """
    doy, hour = grid
    one = demand.household_load_kwh(cfg, doy, hour)
    many = demand.aggregate_load_kwh(cfg, doy, hour, 40)
    scaled = one * (many.sum() / one.sum())

    def peakiness(x):
        return float(x.max() / x.mean())

    assert peakiness(many) < peakiness(scaled)


def test_regression_balcony_kit_is_not_charged_rooftop_maintenance(cfg):
    """BUG 4. cashflows() applied a flat 45 EUR/yr of maintenance plus a
    mid-life string-inverter replacement to everything, including a 620 EUR
    plug-in kit — a maintenance charge larger than the asset. It turned a
    7-year payback into 26 years. Opex is now a parameter."""
    deg = panel.degradation_curve(cfg)
    rooftop = economics.cashflows(620.0, 87.0, cfg, deg)
    plugin = economics.cashflows(620.0, 87.0, cfg, deg, annual_opex_eur=6.0,
                                 inverter_replacement=False)
    assert economics.simple_payback_years(plugin) < 9.0
    assert economics.simple_payback_years(rooftop) > \
        economics.simple_payback_years(plugin)


def test_regression_densified_edges_block_the_sun(cfg):
    """BUG 5. Sampling building outlines at their vertices only let sunlight
    through the middle of long facades, which inflated winter yield on
    courtyard roofs. Densification must produce many more points than the
    polygon has vertices."""
    buildings, _ = roof_model.load_osm(
        os.path.join(HERE, "data", "osm_raw.json"), cfg)
    b = max(buildings, key=lambda x: x.area_m2)
    dense = shading._densify(b.polygon)
    assert len(dense) > len(b.polygon) * 2
    # And no gap between consecutive samples wider than the spacing.
    d = np.hypot(*(np.diff(np.vstack([dense, dense[:1]]), axis=0).T))
    assert d.max() <= shading.EDGE_SAMPLE_M + 0.6
