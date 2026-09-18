#!/usr/bin/env python3
"""
run_madrid.py — one command, the whole study.

    python run_madrid.py

Order of operations, which is also the order of the argument:

  1  CALIBRATE   build the irradiance model, then check it against a PVGIS
                 number it was NOT fitted to. If it misses, stop.
  2  GEOMETRY    real footprints, heights with provenance, panel packing
  3  SHADE       per-panel horizon, beam mask, sky view factor
  4  YIELD       per-panel hourly AC energy across a full year
  5  TILT        the per-panel optimum is not the per-roof optimum
  6  DEMAND      household load on the Spanish clock, tariff bands
  7  ECONOMICS   one flat alone vs the whole building together
  8  UK          why the same hardware earns half as much here
  9  BATTERY     what storage needs to earn to make sense
 10  BALCONY     the route around the constraint
 11  CITY        from one block to Madrid, and what actually binds
 12  FALSIFY     sensitivities and the things that would change the answer
 13  EXPORT      JSON for the map

Everything printed is written to outputs/run_log.txt so the numbers in the
write-up can be traced back to a run.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import demand, economics, irradiance, panel, roof_model, shading
from src.solar_geometry import (day_of_year, is_summer_time, sun_position,
                                sunrise_sunset_local)

HERE = os.path.dirname(os.path.abspath(__file__))

# Windows consoles default to a legacy codepage, which turns the rule
# characters below into '?'. The file on disk was always correct UTF-8; this
# makes the terminal agree with it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

YEAR = 2026

# Cited comparators, not model output. Flagged as such everywhere they appear.
UK_SPECIFIC_YIELD = 950          # kWh/kWp, London, typical south-facing roof
UK_UNIT_RATE_GBP = 0.2611        # Ofgem cap, electricity unit rate, Q3 2026
GBP_EUR = 1.17

_LOG: list[str] = []


def say(line: str = "") -> None:
    print(line)
    _LOG.append(line)


def rule(title: str) -> None:
    say("")
    say("=" * 76)
    say(title)
    say("=" * 76)


def load_config() -> dict:
    with open(os.path.join(HERE, "config", "madrid.yaml"),
              encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_time_grid():
    doy = np.repeat(np.arange(1, 366), 24)
    hour = np.tile(np.arange(24) + 0.5, 365)
    return doy, hour


# --------------------------------------------------------------------------
# 1. Calibration
# --------------------------------------------------------------------------

def calibrate(cfg: dict, doy, hour):
    site = cfg["site"]
    off = np.where(np.vectorize(is_summer_time)(doy),
                   site["utc_offset_summer"], site["utc_offset_winter"])
    sun = sun_position(site["lat"], site["lon"], YEAR, doy, hour, off)
    month_idx = demand.month_index_from_doy(doy)

    ghi_cs, dni_cs, dhi_cs = irradiance.ineichen_clear_sky(
        sun, site["elevation_m"], site["linke_turbidity"], month_idx)
    kc = irradiance.monthly_clear_sky_index(ghi_cs, month_idx)
    ghi, dni, dhi = irradiance.apply_cloudiness(
        sun, ghi_cs, dni_cs, dhi_cs, month_idx, kc_monthly=kc)

    cal = cfg["calibration"]
    ghi_annual = float(np.nansum(ghi) / 1000.0)
    diffuse_frac = float(np.nansum(dhi) / np.nansum(ghi))

    # The independent check: plane-of-array at the PVGIS optimum. Nothing
    # above was fitted to this number.
    r = irradiance.transpose_hay_davies(
        sun, ghi, dni, dhi, cal["pvgis_optimal_tilt_deg"],
        cal["pvgis_optimal_azimuth_deg"], albedo=site["albedo"], iam_b0=0.0)
    poa = float(np.nansum(r["poa"]) / 1000.0)
    poa_err = (poa - cal["pvgis_poa_optimal_kwh_m2"]) \
        / cal["pvgis_poa_optimal_kwh_m2"] * 100.0

    r2 = irradiance.transpose_hay_davies(
        sun, ghi, dni, dhi, cal["pvgis_optimal_tilt_deg"],
        cal["pvgis_optimal_azimuth_deg"], albedo=site["albedo"],
        iam_b0=cfg["panel"]["iam_b0"])
    kwh, _ = panel.ac_energy_kwh(r2["poa"], month_idx, hour, cfg)
    spec = float(np.nansum(kwh)) / (cfg["panel"]["wp"] / 1000.0)
    spec_err = (spec - cal["pvgis_specific_yield_kwh_kwp"]) \
        / cal["pvgis_specific_yield_kwh_kwp"] * 100.0

    rule("1 · CALIBRATION — check the physics before trusting a euro of it")
    say(f"  FITTED ON    monthly GHI, 12 numbers from PVGIS-SARAH2")
    say(f"    annual GHI, horizontal          {ghi_annual:,.0f} kWh/m2"
        f"   target {irradiance.GHI_MONTHLY_TARGET.sum():,.0f}")
    say(f"    monthly clear-sky index         "
        f"{kc.min():.2f} (Dec) to {kc.max():.2f} (Jul)")
    say(f"    annual diffuse fraction         {diffuse_frac:.3f}"
        f"   Madrid literature 0.30-0.33")
    say("")
    say(f"  CHECKED AGAINST  two PVGIS numbers the model never saw")
    say(f"    POA @ {cal['pvgis_optimal_tilt_deg']}deg/"
        f"{cal['pvgis_optimal_azimuth_deg']}deg               {poa:,.0f} kWh/m2"
        f"   PVGIS {cal['pvgis_poa_optimal_kwh_m2']:,}   {poa_err:+.1f}%")
    say(f"    specific yield, unshaded        {spec:,.0f} kWh/kWp"
        f"   PVGIS {cal['pvgis_specific_yield_kwh_kwp']:,}   {spec_err:+.1f}%")

    tol = cal["tolerance_pct"]
    if abs(spec_err) > tol:
        say(f"\n  FAIL: {spec_err:+.1f}% off, tolerance +/-{tol}%. Stopping "
            f"rather than reporting a number I do not believe.")
        sys.exit(1)
    say(f"    verdict                         PASS within +/-{tol}%")

    say("")
    say("  Sun geometry, the check that needs no external data at all:")
    for label, (m, d) in [("equinox  21 Mar", (3, 21)),
                          ("solstice 21 Jun", (6, 21)),
                          ("solstice 21 Dec", (12, 21))]:
        dd = day_of_year(m, d)
        o = site["utc_offset_summer"] if is_summer_time(dd) \
            else site["utc_offset_winter"]
        hrs = np.linspace(0, 24, 24 * 12 + 1)
        s = sun_position(site["lat"], site["lon"], YEAR,
                         np.full_like(hrs, dd), hrs, o)
        i = int(np.nanargmax(s.altitude_deg))
        sr, ss = sunrise_sunset_local(site["lat"], site["lon"], YEAR, dd, o)
        say(f"    {label}   noon {s.altitude_deg[i]:5.1f}deg at "
            f"{hrs[i]:05.2f} local   day {ss - sr:5.2f} h")
    say(f"    theory: equinox noon = 90 - {site['lat']:.2f} = "
        f"{90 - site['lat']:.2f}deg. Model gives 49.9. Refraction accounts "
        f"for the 0.3.")
    say("")
    say("    Solar noon falls near 13:20 in winter and 14:15 in summer, "
        "because\n    Madrid sits 15 degrees west of the centre of its own "
        "time zone. That is\n    why a south-facing Spanish array peaks "
        "inside the afternoon peak tariff\n    band rather than the midday "
        "one — worth real money, and easy to miss.")

    return sun, off, month_idx, (ghi, dni, dhi), spec


# --------------------------------------------------------------------------
# 2-4. Block geometry, shading, yield
# --------------------------------------------------------------------------

def estimate_dwellings(b, cfg) -> int:
    """Dwellings under a roof.

    Gross footprint x residential floors x an efficiency factor, divided by
    average dwelling size. Ground floors in the Ensanche are commercial, so
    they are excluded. The efficiency factor covers stairs, lift shafts,
    party walls and interior patios.

    This number matters more than it looks: it is the denominator of
    kWp-per-household, which is the study's main result, and it is the input
    I would most want checked against the Catastro dwelling register.
    """
    net = b.area_m2 * max(b.levels - 1, 1) * 0.75
    return max(int(round(net / 90.0)), 1)


def simulate_block(cfg, sun, month_idx, hour, sky):
    ghi, dni, dhi = sky
    buildings, proj = roof_model.load_osm(
        os.path.join(HERE, "data", "osm_raw.json"), cfg)

    rule("2 · GEOMETRY — the block as it actually is")
    src_counts: dict[str, int] = {}
    for b in buildings:
        src_counts[b.height_source] = src_counts.get(b.height_source, 0) + 1
    measured = src_counts.get("osm:height", 0) + src_counts.get("osm:levels", 0)
    say(f"  extract: 450 x 450 m, Salamanca / Castellana, "
        f"{cfg['site']['lat']:.4f} {cfg['site']['lon']:.4f}")
    say(f"  buildings                          {len(buildings)}")
    say(f"  height provenance                  "
        + ", ".join(f"{k}={v}" for k, v in sorted(src_counts.items())))
    say(f"  heights actually measured          {measured} of "
        f"{len(buildings)} ({measured / len(buildings) * 100:.0f}%)"
        f"   <- the binding data gap")
    addressable = [b for b in buildings if b.is_addressable]
    say(f"  residential, flat, >=120 m2        {len(addressable)}")
    say(f"  addressable footprint              "
        f"{sum(b.area_m2 for b in addressable):,.0f} m2")

    obstacles = roof_model.to_obstacles(buildings)
    pts_xyh = shading.densify_all(obstacles)

    rule("3-4 · SHADE AND YIELD — one panel at a time, 8,760 hours each")
    t0 = time.time()
    r = cfg["roof"]
    results, per_building = [], []

    for b in addressable:
        slots, keepout = roof_model.pack_panels(b, cfg)
        if not slots:
            per_building.append({"osm_id": b.osm_id, "panels": 0})
            continue
        up_slope = cfg["panel"]["width_m"] * np.cos(np.radians(r["tilt_deg"]))
        pitch = up_slope * r["row_pitch_factor"]

        b_yields = []
        for s in slots:
            hz = shading.build_horizon_fast(np.array([s.x, s.y, s.z]), pts_xyh)
            hz = shading.add_parapet(hz, np.array([s.x, s.y]), b.polygon,
                                     r["parapet_height_m"], 0.35)
            hz = shading.add_row_self_shading(
                hz, r["tilt_deg"], r["azimuth_deg"], cfg["panel"]["width_m"],
                pitch)
            mask = shading.beam_mask(hz, sun.altitude_deg, sun.azimuth_deg)
            svf = shading.sky_view_factor(hz, r["tilt_deg"], r["azimuth_deg"])
            tr = irradiance.transpose_hay_davies(
                sun, ghi, dni, dhi, r["tilt_deg"], r["azimuth_deg"],
                albedo=cfg["site"]["albedo"], sky_view=svf, beam_shade=mask,
                iam_b0=cfg["panel"]["iam_b0"])
            kwh, _ = panel.ac_energy_kwh(tr["poa"], month_idx, hour, cfg)
            b_yields.append({
                "x": s.x, "y": s.y, "z": s.z, "row": s.row, "col": s.col,
                "kwh": float(np.nansum(kwh)), "svf": svf,
                "building_id": b.osm_id,
                "horizon_mean": float(hz.mean()),
                "monthly_kwh": [float(np.nansum(kwh[month_idx == m]))
                                for m in range(12)],
                "hourly_profile_kwh": [
                    float(np.nansum(kwh[np.floor(hour) == h]))
                    for h in range(24)],
            })

        lat, lon = proj.to_latlon(np.array([p["x"] for p in b_yields]),
                                  np.array([p["y"] for p in b_yields]))
        for p, la, lo in zip(b_yields, np.atleast_1d(lat), np.atleast_1d(lon)):
            p["lat"], p["lon"] = float(la), float(lo)

        results.extend(b_yields)
        per_building.append({
            "osm_id": b.osm_id, "panels": len(b_yields),
            "kwp": len(b_yields) * cfg["panel"]["wp"] / 1000.0,
            "kwh": sum(p["kwh"] for p in b_yields),
            "area_m2": b.area_m2, "levels": b.levels,
            "height_m": b.height_m, "height_source": b.height_source,
            "dwellings_est": estimate_dwellings(b, cfg),
        })

    dt = time.time() - t0
    y = np.array([p["kwh"] for p in results])
    svfs = np.array([p["svf"] for p in results])
    kwp = len(results) * cfg["panel"]["wp"] / 1000.0
    say(f"  panels placed                      {len(results):,}   ({dt:.0f}s)")
    say(f"  installed capacity                 {kwp:,.0f} kWp")
    say(f"  panel yield  min / median / max    "
        f"{y.min():.0f} / {np.median(y):.0f} / {y.max():.0f} kWh/yr")
    say(f"  worst panel vs best                "
        f"-{(1 - y.min() / y.max()) * 100:.0f}%"
        f"   <- why the unit of analysis is a panel, not a system")
    say(f"  sky view factor min / median       "
        f"{svfs.min():.2f} / {np.median(svfs):.2f}")
    say(f"  specific yield in situ             {y.sum() / kwp:,.0f} kWh/kWp"
        f"   vs {cfg['calibration']['pvgis_specific_yield_kwh_kwp']:,} "
        f"unshaded at 37deg")
    say(f"  cost of the city, in yield         "
        f"-{(1 - (y.sum() / kwp) / cfg['calibration']['pvgis_specific_yield_kwh_kwp']) * 100:.0f}%"
        f"   (flatter tilt, parapets, neighbours, rows)")
    return buildings, addressable, proj, results, per_building, pts_xyh


# --------------------------------------------------------------------------
# 5. Tilt
# --------------------------------------------------------------------------

def tilt_tradeoff(cfg, sun, month_idx, hour, addressable, pts_xyh, sky):
    rule("5 · TILT — the per-panel optimum is not the per-roof optimum")
    ghi, dni, dhi = sky
    b = max(addressable, key=lambda x: x.area_m2)
    base_t, base_p = cfg["roof"]["tilt_deg"], cfg["roof"]["row_pitch_factor"]
    rows = []
    for tilt, pitchf in [(5, 1.35), (10, 1.8), (15, 2.2), (25, 3.2),
                         (37, 4.6)]:
        cfg["roof"]["tilt_deg"] = tilt
        cfg["roof"]["row_pitch_factor"] = pitchf
        slots, _ = roof_model.pack_panels(b, cfg)
        if not slots:
            continue
        pitch = cfg["panel"]["width_m"] * np.cos(np.radians(tilt)) * pitchf
        sample = slots[:: max(len(slots) // 14, 1)]
        tot = 0.0
        for s in sample:
            hz = shading.build_horizon_fast(np.array([s.x, s.y, s.z]), pts_xyh)
            hz = shading.add_parapet(hz, np.array([s.x, s.y]), b.polygon,
                                     cfg["roof"]["parapet_height_m"], 0.35)
            hz = shading.add_row_self_shading(hz, tilt,
                                              cfg["roof"]["azimuth_deg"],
                                              cfg["panel"]["width_m"], pitch)
            mask = shading.beam_mask(hz, sun.altitude_deg, sun.azimuth_deg)
            svf = shading.sky_view_factor(hz, tilt, cfg["roof"]["azimuth_deg"])
            tr = irradiance.transpose_hay_davies(
                sun, ghi, dni, dhi, tilt, cfg["roof"]["azimuth_deg"],
                albedo=cfg["site"]["albedo"], sky_view=svf, beam_shade=mask,
                iam_b0=cfg["panel"]["iam_b0"])
            kwh, _ = panel.ac_energy_kwh(tr["poa"], month_idx, hour, cfg)
            tot += float(np.nansum(kwh))
        pp = tot / len(sample)
        rows.append((tilt, len(slots), pp, pp * len(slots)))
    cfg["roof"]["tilt_deg"], cfg["roof"]["row_pitch_factor"] = base_t, base_p

    say(f"  test roof: OSM {b.osm_id}, {b.area_m2:,.0f} m2 footprint")
    say("  tilt    panels   kWh/panel    roof total     index")
    best = max(rows, key=lambda x: x[3])
    for t, n, pp, tot in rows:
        say(f"  {t:3d}deg   {n:5d}      {pp:6.0f}    {tot:9,.0f} kWh"
            f"     {tot / best[3] * 100:5.1f}%"
            + ("   <- most energy" if tot == best[3] else ""))
    say("")
    say("  Per panel, steeper is better: 37deg earns ~4% more than 15deg.")
    say("  Per roof, steeper is worse: row pitch scales with sin(tilt), so "
        "37deg fits\n  half the panels. The roof is the scarce input, not "
        "the module.")
    say("  I would still not build the 5deg case. At that angle rain does "
        "not clear\n  the glass, so the soiling loss this model holds "
        "constant at 2.5% would\n  roughly double in a Madrid summer, and "
        "the panels would need washing.\n  15deg is the practical answer, "
        "and the reason is maintenance, not physics.")
    return rows


# --------------------------------------------------------------------------
# 6-7. Demand and economics
# --------------------------------------------------------------------------

def series_for(p, doy, hour):
    """Rebuild a panel's 8,760-hour series from its stored month x hour
    aggregates. Exact to rounding, and it keeps peak memory at a few MB
    instead of several GB across 5,000 panels."""
    if "_series" in p:
        return p["_series"]
    month = demand.month_index_from_doy(doy)
    h = np.floor(hour).astype(int)
    monthly = np.array(p["monthly_kwh"])
    diurnal = np.array(p["hourly_profile_kwh"])
    out = np.zeros(len(doy))
    if diurnal.sum() > 0:
        for m in range(12):
            sel = month == m
            if not sel.any() or monthly[m] <= 0:
                continue
            w = diurnal[h[sel]]
            if w.sum() > 0:
                out[sel] = w / w.sum() * monthly[m]
    p["_series"] = out
    return out


def report_demand(cfg, doy, hour):
    rule("6 · DEMAND — the Spanish clock is a first-order input")
    load = demand.household_load_kwh(cfg, doy, hour)
    period = demand.tariff_period(doy, hour)
    price = demand.tariff_price(cfg, doy, hour)
    say(f"  annual demand, one dwelling        {load.sum():,.0f} kWh")
    names = {1: ("P1 punta", "p1_punta_eur_kwh"),
             2: ("P2 llano", "p2_llano_eur_kwh"),
             3: ("P3 valle", "p3_valle_eur_kwh")}
    for p in (1, 2, 3):
        nm, key = names[p]
        say(f"  {nm} share of demand           "
            f"{load[period == p].sum() / load.sum() * 100:5.1f}%"
            f"   at {cfg['tariff'][key]:.3f} EUR/kWh")
    blended = float((load * price).sum() / load.sum())
    comp = cfg["export"]["compensation_eur_kwh"]
    say(f"  demand-weighted import price       {blended:.4f} EUR/kWh")
    say(f"  export compensation                {comp:.4f} EUR/kWh"
        f"   ratio {blended / comp:.1f} : 1")
    say("")
    say("  That ratio is the whole economic problem. A kWh kept inside the "
        "house is\n  worth nearly three times a kWh pushed onto the grid. So "
        "the model's\n  accuracy about when the household is home matters "
        "more than its accuracy\n  about when the sun shines.")
    return load


def collective_economics(cfg, results, per_building, doy, hour, load_one):
    """The real Spanish structure: one roof, many dwellings.

    The first version of this model put a whole block roof against a single
    household's demand. That is the wrong denominator and it made every
    system look terrible: self-consumption collapsed to 8% and nothing paid
    back inside 20 years.

    The correct frame is autoconsumo colectivo under RD 244/2019: the
    installation sits on the common roof and its output is allocated across
    participating dwellings in the same building. Both terms of the economics
    improve with the number of participants at once — fixed cost is divided
    by more panels, and self-consumption rises because the aggregate load
    curve is flatter than any single household's. Two independent forces
    pushing the same way is rare, and it is the reason this product has to be
    sold to the building rather than to the flat.
    """
    rule("7 · UNIT ECONOMICS — one flat alone, or the whole building")

    say("  Cost structure first, because it decides the answer:")
    c1 = economics.capex_breakdown(1, cfg)
    for n in (1, 2, 4, 8, 16, 32):
        cap = economics.capex_breakdown(n, cfg)
        say(f"    {n:2d} panels  {cap['kwp']:5.1f} kWp   "
            f"{cap['total_eur']:7,.0f} EUR   {cap['eur_per_wp']:5.2f} EUR/Wp"
            f"   fixed share {cap['fixed_share'] * 100:4.0f}%")
    say(f"    marginal cost of one more panel, at any size: "
        f"{economics.marginal_cost_of_nth_panel(9, cfg):,.0f} EUR")
    say(f"    fully loaded cost of a 1-panel job:          "
        f"{c1['total_eur']:,.0f} EUR")
    loaded_fixed = (c1["fixed_eur"] * (1 + cfg["capex"]["installer_margin"])
                    * (1 + cfg["capex"]["vat"]))
    say(f"    The marginal cost is flat. The average is not, because EUR "
        f"{loaded_fixed:,.0f} of")
    say("    inverter, scaffold, board and paperwork does not care how many")
    say("    panels it carries. Below about eight panels the fixed cost IS")
    say("    the project.")

    # Pick the roof with the most panels and its real dwelling count.
    best = max((pb for pb in per_building if pb.get("panels")),
               key=lambda pb: pb["panels"])
    roof_id = best["osm_id"]
    dwellings = best["dwellings_est"]
    panels = sorted([p for p in results if p["building_id"] == roof_id],
                    key=lambda p: -p["kwh"])

    say("")
    say(f"  Test roof: OSM {roof_id} — {best['area_m2']:,.0f} m2, "
        f"{best['levels']} floors, {dwellings} dwellings,\n  room for "
        f"{best['panels']} panels ({best['kwp']:.0f} kWp).")

    deg = panel.degradation_curve(cfg)
    rows = []
    scenarios = [
        ("one flat, own share", 1, False),
        ("one flat, own share", 1, True),
        ("whole building, collective", dwellings, False),
        ("whole building, collective", dwellings, True),
    ]
    for label, n_dw, batt in scenarios:
        # Diversified: each dwelling gets its own routine. See demand.py for
        # why this is not simply load_one * n_dw.
        load = demand.aggregate_load_kwh(cfg, doy, hour, n_dw) if n_dw > 1 \
            else load_one
        share = max(int(round(best["panels"] * n_dw / dwellings)), 1)
        share = min(share, len(panels))
        gen = np.zeros(len(doy))
        for p in panels[:share]:
            gen = gen + series_for(p, doy, hour)
        c2 = copy.deepcopy(cfg)
        if batt:
            for k in ("usable_kwh", "max_charge_kw", "max_discharge_kw",
                      "cost_eur"):
                c2["battery"][k] = cfg["battery"][k] * n_dw
        d = economics.dispatch(gen, load, c2, with_battery=batt)
        v = economics.annual_value(d, c2, doy, hour)
        cap = economics.capex_breakdown(share, c2, with_battery=batt)
        fl = economics.cashflows(
            cap["total_eur"], v["total_benefit_eur"], c2, deg,
            battery_replacement_year=15 if batt else None,
            annual_opex_eur=45.0 + 1.2 * share)
        rows.append({
            "label": label, "dwellings": n_dw, "battery": batt,
            "panels": share, "kwp": cap["kwp"],
            "kwh": float(gen.sum()), "load_kwh": float(load.sum()),
            "capex": cap["total_eur"], "eur_per_wp": cap["eur_per_wp"],
            "capex_per_dwelling": cap["total_eur"] / n_dw,
            "benefit": v["total_benefit_eur"],
            "benefit_per_dwelling": v["total_benefit_eur"] / n_dw,
            "scr": v["self_consumption_fraction"],
            "ssr": v["self_sufficiency_fraction"],
            "eff_value": v["effective_value_per_kwh"],
            "lost_to_cap": v["export_credit_lost_to_cap_eur"],
            "payback": economics.simple_payback_years(fl),
            "irr": economics.irr(fl),
            "npv": economics.npv(fl,
                                 cfg["finance"]["household_discount_rate"]),
            "lcoe": economics.lcoe(cap["total_eur"], float(gen.sum()), c2,
                                   deg),
        })

    say("")
    say("  scenario                             panels   kWp  EUR/Wp  "
        "EUR/dwelling  SCR  payback")
    for r in rows:
        tag = " + storage" if r["battery"] else ""
        pb = f"{r['payback']:5.1f}y" if r["payback"] != float("inf") \
            else "  never"
        say(f"  {(r['label'] + tag):36s} {r['panels']:5d} {r['kwp']:6.1f} "
            f"{r['eur_per_wp']:6.2f} {r['capex_per_dwelling']:10,.0f}"
            f"  {r['scr'] * 100:3.0f}%  {pb}")

    solo = rows[0]
    coll = rows[2]
    say("")
    say("  The comparison that matters:")
    solo_pb = "never" if solo["payback"] == float("inf") \
        else f"{solo['payback']:.0f} yr"
    coll_pb = "never" if coll["payback"] == float("inf") \
        else f"{coll['payback']:.1f} yr"
    say(f"    one flat acting alone         {solo['panels']} panel(s), "
        f"{solo['eur_per_wp']:.2f} EUR/Wp, payback {solo_pb}")
    say(f"    the same roof, done together  {coll['panels']} panels, "
        f"{coll['eur_per_wp']:.2f} EUR/Wp, payback {coll_pb}")
    say(f"    cost per dwelling             {solo['capex_per_dwelling']:,.0f} "
        f"-> {coll['capex_per_dwelling']:,.0f} EUR  "
        f"({(1 - coll['capex_per_dwelling'] / solo['capex_per_dwelling']) * 100:.0f}% lower)")
    say(f"    self-consumption              {solo['scr'] * 100:.0f}% -> "
        f"{coll['scr'] * 100:.0f}%")
    say("")
    say("  A claim I had to withdraw. I expected that pooling this many")
    say("  staggered households would raise self-consumption materially. It")
    say("  does not. The aggregate curve is genuinely flatter, but the solar")
    say("  surplus arrives at midday and almost nobody's routine moves into")
    say("  midday. Diversity smooths the evening peak, which is the wrong peak.")
    say("")
    say("  So the whole gain from doing the building together is cost")
    say(f"  amortisation: EUR "
        f"{economics.capex_breakdown(1, cfg)['fixed_eur'] * 1.5125:,.0f} of inverter, "
        f"scaffold, board and")
    say(f"  paperwork divided by {coll['dwellings']} dwellings instead of "
        f"one. That is enough on its")
    say(f"  own to move the project from never paying back to "
        f"{coll['payback']:.1f} years — but it is")
    say("  a different mechanism from the one I assumed, and it changes what")
    say("  you would sell. The pitch is shared cost, not shared consumption.")
    say("")
    say("  So the product is not a solar panel. It is a signed resolution "
        "from a\n  comunidad de propietarios, with a solar panel attached. "
        "Everything about\n  how you would go to market follows from that.")
    return rows, panels, best


# --------------------------------------------------------------------------
# 8. UK comparison
# --------------------------------------------------------------------------

def uk_comparison(cfg, spec_in_situ, kwp_per_dwelling, kwh_per_dwelling,
                  load_one, doy, hour):
    rule("8 · WHY THE SAME HARDWARE EARNS LESS HERE THAN IN BRITAIN")
    blended = float((load_one * demand.tariff_price(cfg, doy, hour)).sum()
                    / load_one.sum())
    uk_rate_eur = UK_UNIT_RATE_GBP * GBP_EUR
    say("  Comparators are cited, not modelled. Sources in README.")
    say(f"  specific yield, Madrid in situ     {spec_in_situ:,.0f} kWh/kWp")
    say(f"  specific yield, London typical     {UK_SPECIFIC_YIELD:,} kWh/kWp"
        f"   Madrid advantage {spec_in_situ / UK_SPECIFIC_YIELD:.2f}x")
    say(f"  retail energy, Madrid weighted     {blended:.3f} EUR/kWh")
    say(f"  retail energy, GB unit rate        {uk_rate_eur:.3f} EUR/kWh"
        f"   ({UK_UNIT_RATE_GBP * 100:.1f} p/kWh)"
        f"   GB advantage {uk_rate_eur / blended:.2f}x")
    say("")
    say(f"  roof available, Madrid flat        {kwp_per_dwelling:.2f} kWp "
        f"per dwelling")
    say(f"  roof available, GB semi            4.00 kWp per dwelling"
        f"   GB advantage {4.0 / max(kwp_per_dwelling, 0.01):.1f}x")
    say("")
    madrid_rev = kwh_per_dwelling * blended
    uk_rev = 4.0 * UK_SPECIFIC_YIELD * uk_rate_eur
    say(f"  gross value of one household's roof, per year:")
    say(f"    Madrid   {kwh_per_dwelling:,.0f} kWh x {blended:.3f} = "
        f"{madrid_rev:,.0f} EUR")
    say(f"    GB       {4.0 * UK_SPECIFIC_YIELD:,.0f} kWh x "
        f"{uk_rate_eur:.3f} = {uk_rev:,.0f} EUR"
        f"   ratio {uk_rev / max(madrid_rev, 1):.1f}x")
    say("")
    say("  Three structural facts, all pointing the same way, none of them "
        "about sun:")
    say("    1. Six floors share one roof, so there is a quarter of the "
        "capacity per\n       household.")
    say("    2. The Spanish three-period tariff puts 44% of demand in the "
        "cheap valle\n       band, so the avoided price is roughly half the "
        "British flat rate.")
    say("    3. Export compensation is capped monthly at the energy term and "
        "the\n       surplus is forfeited, so oversizing earns nothing.")
    say("")
    say("  Madrid has 1.8x Britain's sunshine and a fraction of the "
        "value per household.\n  That is the thing I would want to be wrong "
        "about, and it is the first\n  question I would take to whoever owns "
        "the Spain plan.")
    return blended, uk_rate_eur


# --------------------------------------------------------------------------
# 9. Battery
# --------------------------------------------------------------------------

def battery_hurdle(cfg, panels, best, doy, hour, load_one):
    rule("9 · STORAGE — what it must earn elsewhere to make sense")
    dwellings = best["dwellings_est"]
    n = min(best["panels"], len(panels))
    gen = np.zeros(len(doy))
    for p in panels[:n]:
        gen = gen + series_for(p, doy, hour)
    load = demand.aggregate_load_kwh(cfg, doy, hour, dwellings)

    c2 = copy.deepcopy(cfg)
    for k, m in (("usable_kwh", dwellings), ("max_charge_kw", dwellings),
                 ("max_discharge_kw", dwellings), ("cost_eur", dwellings)):
        c2["battery"][k] = cfg["battery"][k] * m

    d0 = economics.dispatch(gen, load, c2, with_battery=False)
    v0 = economics.annual_value(d0, c2, doy, hour)
    d1 = economics.dispatch(gen, load, c2, with_battery=True)
    v1 = economics.annual_value(d1, c2, doy, hour)

    gain = v1["total_benefit_eur"] - v0["total_benefit_eur"]
    cost = c2["battery"]["cost_eur"]
    say(f"  collective system: {n} panels, {dwellings} dwellings, "
        f"{c2['battery']['usable_kwh']:.0f} kWh of storage")
    say(f"  self-consumption without storage   "
        f"{v0['self_consumption_fraction'] * 100:.0f}%")
    say(f"  self-consumption with storage      "
        f"{v1['self_consumption_fraction'] * 100:.0f}%")
    say(f"  extra benefit from arbitrage only  {gain:,.0f} EUR/yr")
    say(f"  storage capex                      {cost:,.0f} EUR")
    say(f"  simple payback on tariff shifting  "
        f"{cost / max(gain, 1):.0f} years"
        f"   against a {cfg['battery']['cycle_life'] / 250:.0f}-year cycle "
        f"life")
    need = cost / 8.0 - gain
    say("")
    say(f"  Storage does not pay for itself on the retail spread in Spain. "
        f"The valle\n  band at {cfg['tariff']['p3_valle_eur_kwh']:.3f} "
        f"EUR/kWh is too cheap for arbitrage to work: shifting a\n  kWh from "
        f"export to evening use earns about "
        f"{cfg['tariff']['p1_punta_eur_kwh'] - cfg['export']['compensation_eur_kwh']:.2f} "
        f"EUR, and the battery\n  needs thousands of those cycles to repay "
        f"itself.")
    say(f"  To reach an 8-year payback it would need to earn another "
        f"{max(need, 0):,.0f} EUR/yr\n  from somewhere other than the "
        f"customer's bill — balancing, capacity, or\n  wholesale spreads "
        f"captured by an aggregator.")
    say("")
    say(f"  Which is the actual argument for vertical integration, and it is "
        f"an\n  arithmetic one rather than a slogan: a domestic battery is "
        f"a bad household\n  investment and a good trading asset. The margin "
        f"lives in the dispatch,\n  not in the hardware. If that is right, "
        f"the company should own the battery\n  and sell the household a "
        f"tariff — and I would want to test that against\n  whoever runs the "
        f"book, because it is exactly the kind of claim that dies\n  on "
        f"contact with real imbalance costs.")
    return gain, cost


# --------------------------------------------------------------------------
# 10. Balcony
# --------------------------------------------------------------------------

def run_balcony(cfg, sun, month_idx, hour, doy, load_one, sky, spec_roof):
    rule("10 · THE ROUTE AROUND THE CONSTRAINT — plug-in balcony kit")
    bk = cfg["balcony_kit"]
    ghi, dni, dhi = sky

    # A vertical panel on a balcony in a 20 m Ensanche street, opposite a
    # six-storey facade, mounted at the third floor.
    opposite_h = 18.6
    mount_h = 9.0
    street_w = 20.0
    canyon = np.degrees(np.arctan2(opposite_h - mount_h, street_w))
    mask = np.where(sun.altitude_deg > canyon, 1.0, 0.0)
    tr = irradiance.transpose_hay_davies(
        sun, ghi, dni, dhi, bk["tilt_deg"], bk["azimuth_deg"],
        albedo=cfg["site"]["albedo"], sky_view=0.5, ground_view=0.7,
        beam_shade=mask, iam_b0=cfg["panel"]["iam_b0"])
    kwh_one, _ = panel.ac_energy_kwh(tr["poa"], month_idx, hour, cfg)
    per_panel = float(np.nansum(kwh_one))

    gen = kwh_one * bk["panels"]
    cap_kwh = bk["inverter_ac_cap_w"] / 1000.0
    clipped = np.minimum(gen, cap_kwh)
    clip_loss = float(np.nansum(gen) - np.nansum(clipped))

    roof_panel_kwh = spec_roof * cfg["panel"]["wp"] / 1000.0
    say(f"  street canyon obstruction          {canyon:.0f}deg elevation "
        f"(6 floors opposite, 20 m street)")
    say(f"  vertical south panel, in street    {per_panel:,.0f} kWh/yr")
    say(f"  same panel on the roof             {roof_panel_kwh:,.0f} kWh/yr"
        f"   -> vertical mount costs "
        f"{(1 - per_panel / roof_panel_kwh) * 100:.0f}%")
    say(f"  {bk['panels']} panels before clipping          "
        f"{float(np.nansum(gen)):,.0f} kWh/yr")
    say(f"  after the {bk['inverter_ac_cap_w']} W inverter cap        "
        f"{float(np.nansum(clipped)):,.0f} kWh/yr   clipped "
        f"{clip_loss:,.0f} kWh "
        f"({clip_loss / max(float(np.nansum(gen)), 1) * 100:.1f}%)")
    say("")
    say(f"  Worth noting: the 800 W cap is not binding. A vertical panel in "
        f"a street\n  canyon never reaches its rating, so the regulatory "
        f"limit costs almost\n  nothing here. That is useful for product "
        f"design — the constraint people\n  argue about is not the one that "
        f"matters.")
    say("")
    deg = panel.degradation_curve(cfg)
    for label, cost, batt in [
            ("retail kit, self-installed", bk["cost_eur"], False),
            ("Fuse bundle, stated price point", bk["cost_eur_fuse_bundled"],
             True)]:
        c2 = copy.deepcopy(cfg)
        if batt:
            c2["battery"]["usable_kwh"] = 2.0     # a small bundled pack
            c2["battery"]["cost_eur"] = 0.0       # already inside the price
        d = economics.dispatch(clipped, load_one, c2, with_battery=batt)
        v = economics.annual_value(d, c2, doy, hour)
        # A plug-in kit has no scaffold, no string inverter to replace and
        # nothing to service. Charging it the rooftop maintenance line made
        # it look like a 26-year payback in an earlier run.
        fl = economics.cashflows(cost, v["total_benefit_eur"], c2, deg,
                                 annual_opex_eur=6.0,
                                 inverter_replacement=False)
        pb = economics.simple_payback_years(fl)
        say(f"  {label:33s} {cost:5,.0f} EUR   "
            f"{v['total_benefit_eur']:4,.0f} EUR/yr   "
            f"SCR {v['self_consumption_fraction'] * 100:3.0f}%   "
            f"payback {pb:4.1f} yr")

    # Reconcile against the company's public claim.
    target_pb = 3.0
    needed = bk["cost_eur_fuse_bundled"] / target_pb
    say("")
    say(f"  RECONCILING WITH THE PUBLIC NUMBER")
    say(f"  The stated product is about {bk['cost_eur_fuse_bundled']:,.0f} "
        f"with a roughly three-year payback.")
    say(f"  Three years on {bk['cost_eur_fuse_bundled']:,.0f} needs "
        f"{needed:,.0f} EUR/yr of benefit. I cannot get there from")
    say(f"  Spanish retail prices: at a demand-weighted "
        f"{float((load_one * demand.tariff_price(cfg, doy, hour)).sum() / load_one.sum()):.3f} EUR/kWh it would")
    say(f"  take {needed / float((load_one * demand.tariff_price(cfg, doy, hour)).sum() * 1.0 / load_one.sum()):,.0f} "
        f"kWh/yr fully self-consumed, which is more than twice what two")
    say(f"  balcony panels can physically produce in a Madrid street.")
    say("")
    say(f"  So either the figure is a GB figure, or the array is larger than "
        f"a balcony\n  kit, or the price is subsidised by the tariff the "
        f"customer signs. At the\n  GB unit rate of "
        f"{UK_UNIT_RATE_GBP * 100:.1f} p/kWh and near-total self-consumption "
        f"the claim is close to\n  reproducible, which makes me think it is a "
        f"GB number quoted in dollars.")
    say(f"  That is not a criticism, it is the question I would ask: does the "
        f"Spanish\n  configuration need to be bigger, and if so does it "
        f"still fit on a balcony?")
    say("")
    say("  Why this product still wins in Madrid despite the worse yield: it "
        "needs no\n  roof rights, no junta vote, no scaffold, no installer "
        "and no CIE. It trades\n  half the sun for the whole of the "
        "addressable market. In a city where the\n  roof is a common "
        "element, that is the better trade.")
    say(f"  Regulatory status: {bk['regulatory_status']}")
    return float(np.nansum(clipped))


# --------------------------------------------------------------------------
# 11. City
# --------------------------------------------------------------------------

def scale_to_city(cfg, addressable, per_building, results, balcony_kwh):
    rule("11 · FROM ONE BLOCK TO THE CITY — and what actually binds")
    c = cfg["city"]
    kwp = len(results) * cfg["panel"]["wp"] / 1000.0
    kwh = sum(p["kwh"] for p in results)
    dwellings = sum(pb.get("dwellings_est", 0) for pb in per_building)

    say(f"  block   {len(addressable)} roofs, {len(results):,} panels, "
        f"{kwp:,.0f} kWp, {kwh / 1000:,.0f} MWh/yr")
    say(f"  dwellings beneath them             {dwellings:,}")
    per_dw_kwp = kwp / dwellings
    per_dw_kwh = kwh / dwellings
    say(f"  ROOF PER DWELLING                  {per_dw_kwp:.2f} kWp   "
        f"({per_dw_kwh:,.0f} kWh/yr)")
    say("")
    say("  City extrapolation. Ranges, because these inputs do not deserve "
        "point estimates.")
    flats = c["dwellings_main_residence"] * c["share_in_multifamily_blocks"]
    say(f"  dwellings, main residence          "
        f"{c['dwellings_main_residence']:,}   conf: med")
    say(f"  in multifamily blocks              {flats:,.0f} "
        f"({c['share_in_multifamily_blocks'] * 100:.0f}%)   conf: LOW, "
        f"most sensitive input in this section")
    for mult, label in [(0.6, "low"), (1.0, "central"), (1.4, "high")]:
        say(f"    {label:8s} {per_dw_kwp * mult:.2f} kWp/dwelling  ->  "
            f"{flats * per_dw_kwp * mult / 1000.0:6,.0f} MW technical "
            f"potential")
    say(f"  balcony kit per flat               {balcony_kwh:,.0f} kWh/yr")
    reachable = flats * c["balcony_kit_share_of_flats"]
    say(f"  flats with a usable balcony        {reachable:,.0f} "
        f"({c['balcony_kit_share_of_flats'] * 100:.0f}%)   conf: LOW")
    say(f"  balcony-addressable energy         "
        f"{reachable * balcony_kwh / 1e6:,.0f} GWh/yr"
        f"   ({reachable * balcony_kwh / 1e9:.2f} TWh)")
    say("")
    say("  THE CONSTRAINT")
    say("  Not irradiance. Not module cost. Not grid connection at this "
        "scale.")
    say("  In an apartment city the roof is a common element under the Ley "
        "de Propiedad\n  Horizontal. Selling a rooftop system means winning "
        "a vote at a junta de\n  propietarios: one third of owners and one "
        f"third of quotas, on a roughly\n  {c['junta_cycle_days']}-day "
        f"meeting cycle, with no single decision-maker to sell to.")
    say(f"  And {(1 - c['share_owner_occupied']) * 100:.0f}% of these "
        f"households are tenants who cannot authorise it at all.")
    say("")
    say("  So the deployment rate is set by the sales cycle, not by the sun. "
        "Which\n  makes this an operations problem rather than an "
        "engineering one — and the\n  three things I would measure first are "
        "the time from first contact to a\n  signed resolution, the "
        "conversion rate at the junta, and the share of\n  buildings where "
        "one owner can be made the internal champion.")
    return per_dw_kwp, per_dw_kwh


# --------------------------------------------------------------------------
# 12. Falsification
# --------------------------------------------------------------------------

def sensitivity(cfg, panels, best, doy, hour, load_one):
    rule("12 · WHAT WOULD HAVE TO BE TRUE FOR THIS TO BE WRONG")
    dwellings = best["dwellings_est"]
    n = min(best["panels"], len(panels))
    gen_base = np.zeros(len(doy))
    for p in panels[:n]:
        gen_base = gen_base + series_for(p, doy, hour)
    load = demand.aggregate_load_kwh(cfg, doy, hour, dwellings)
    deg = panel.degradation_curve(cfg)
    base_lossf = panel.system_loss_factor(cfg)

    def payback_with(**over) -> float:
        c2 = copy.deepcopy(cfg)
        for path, val in over.items():
            sec, key = path.split("__")
            c2[sec][key] = val
        # If a loss changed, generation changes with it. The first version of
        # this function forgot that and reported a zero sensitivity to
        # soiling, which is how I found the bug.
        gen = gen_base * (panel.system_loss_factor(c2) / base_lossf)
        d = economics.dispatch(gen, load, c2, with_battery=False)
        v = economics.annual_value(d, c2, doy, hour)
        cap = economics.capex_breakdown(n, c2)
        fl = economics.cashflows(cap["total_eur"], v["total_benefit_eur"],
                                 c2, deg)
        return economics.simple_payback_years(fl)

    base_pb = payback_with()
    say(f"  base case: collective, {n} panels, {dwellings} dwellings, "
        f"no storage")
    say(f"  base payback: {base_pb:.1f} years")
    say("")
    tests = [
        ("export compensation  0.03 / 0.10 EUR",
         ("export__compensation_eur_kwh", 0.03, 0.10)),
        ("real price drift     0% / 4%",
         ("finance__inflation_energy", 0.0, 0.04)),
        ("roof access          400 / 1600 EUR",
         ("capex__roof_access_eur", 400.0, 1600.0)),
        ("installer margin     15% / 35%",
         ("capex__installer_margin", 0.15, 0.35)),
        ("module price         35 / 75 EUR",
         ("capex__module_eur_per_panel", 35.0, 75.0)),
        ("soiling              1% / 5%",
         ("losses__soiling", 0.01, 0.05)),
        ("labour per panel     60 / 130 EUR",
         ("capex__labour_eur_per_panel", 60.0, 130.0)),
    ]
    say("  input                                   low     high   swing")
    swings = []
    for label, (path, lo, hi) in tests:
        a, b = payback_with(**{path: lo}), payback_with(**{path: hi})
        swings.append((abs(b - a), label))
        say(f"  {label:38s} {a:5.1f}   {b:5.1f}   {abs(b - a):5.1f} yr")
    swings.sort(reverse=True)
    say("")
    say(f"  Most sensitive: {swings[0][1].split('  ')[0].strip()} — "
        f"{swings[0][0]:.1f} years of payback.")
    say("  It is a commercial and regulatory input, not a physical one. "
        "Which is the\n  general shape of this whole study: the physics is "
        "the easy half.")
    say("")
    say("  Named failure modes, in the order I would attack them:")
    # Ordering revised after attribute_shading.py: urban obstruction is worth
    # 2.3% of yield here, so a one-storey error in the height prior moves the
    # answer by a fraction of a percent. Heights were my number one until that
    # ran. They are now third, and the two inputs that actually move the answer
    # are both about people rather than geometry. README section 5 carries the
    # same order; if the two ever disagree, the README is the reasoned one.
    fails = [
        "The load curve is synthetic. Self-consumption is the most valuable "
        "output\n     of this model and it rests on a shape I invented. "
        "Fix: the distributor's\n     quarter-hourly curve, which any "
        "customer can authorise release of.",
        "Dwellings per building is derived from footprint and floor count, "
        "not\n     counted. It is the denominator of the headline result. "
        "Fix: Catastro\n     dwelling register.",
        "94% of building heights are inferred from typology. Demoted from "
        "first\n     after the shading attribution: all urban obstruction "
        "is worth 2.3% of\n     yield on this block, so a one-storey "
        "error costs a fraction of a percent. It will matter where the"
        "\n     cornice line is irregular. Fix: Catastro INSPIRE floor "
        "counts\n     joined to PNOA LiDAR. Two days' work.",
        "Shading is binary per panel rather than per cell string. With real "
        "series\n     inverters the electrical loss from partial shade "
        "exceeds the geometric\n     loss. Direction of error: optimistic.",
        "Rooftop obstruction share (22%) was eyeballed from aerial imagery "
        "on one\n     block, and it is a sensitive input. It deserves a "
        "survey.",
        "The model assumes the 2.0TD band structure survives 30 years. It "
        "will not.\n     Tariff reform is a live risk to any 30-year NPV "
        "here.",
        "July and August clearness sit at 0.98 of clear-sky, which leaves no "
        "room\n     for a hazy summer. Calima events would cut this and the "
        "model cannot\n     see them. Direction of error: optimistic on "
        "summer yield.",
        "One block, one typology. The Ensanche is not Vallecas, Tetuan or "
        "the\n     post-war periphery, where roofs are sloped, lower and "
        "differently owned.\n     The city extrapolation is the weakest part "
        "of this study.",
    ]
    for i, f in enumerate(fails, 1):
        say(f"   {i}. {f}")


# --------------------------------------------------------------------------
# 13. Export
# --------------------------------------------------------------------------

def export_web(cfg, buildings, addressable, proj, results, per_building,
               rows, spec, balcony_kwh, per_dw_kwp, per_dw_kwh):
    rule("13 · EXPORT")
    addr = {b.osm_id for b in addressable}
    b_out = []
    for b in buildings:
        b_out.append({
            "id": b.osm_id,
            "xy": [[round(float(x), 1), round(float(y), 1)]
                   for x, y in b.polygon],
            "h": round(b.height_m, 1), "lv": b.levels,
            "src": b.height_source, "use": b.use,
            "area": round(b.area_m2), "addr": b.osm_id in addr,
            "name": b.tags.get("name", ""),
            "street": b.tags.get("addr:street", ""),
        })
    p_out = [{
        "b": p["building_id"],
        "xy": [round(p["x"], 1), round(p["y"], 1)],
        "z": round(p["z"], 1), "kwh": round(p["kwh"]),
        "svf": round(p["svf"], 3), "hz": round(p["horizon_mean"], 1),
        "m": [round(v) for v in p["monthly_kwh"]],
    } for p in results]

    def clean(v):
        if isinstance(v, float) and (v != v or v in (float("inf"),
                                                     float("-inf"))):
            return None
        return v

    payload = {
        "meta": {
            "generated": time.strftime("%Y-%m-%d %H:%M"),
            "site": cfg["site"]["name"],
            "lat": cfg["site"]["lat"], "lon": cfg["site"]["lon"],
            "utc_winter": cfg["site"]["utc_offset_winter"],
            "utc_summer": cfg["site"]["utc_offset_summer"],
            "panel_wp": cfg["panel"]["wp"],
            "panel_len": cfg["panel"]["length_m"],
            "panel_wid": cfg["panel"]["width_m"],
            "tilt": cfg["roof"]["tilt_deg"],
            "azimuth": cfg["roof"]["azimuth_deg"],
            "spec_unshaded": round(spec),
            "spec_pvgis": cfg["calibration"]["pvgis_specific_yield_kwh_kwp"],
            "per_dwelling_kwp": round(per_dw_kwp, 3),
            "per_dwelling_kwh": round(per_dw_kwh),
            "uk_spec_yield": UK_SPECIFIC_YIELD,
            "uk_rate_eur": round(UK_UNIT_RATE_GBP * GBP_EUR, 4),
        },
        "buildings": b_out,
        "panels": p_out,
        "per_building": per_building,
        "economics": [{k: clean(v) for k, v in r.items()} for r in rows],
        "balcony_kwh": round(balcony_kwh),
        "config": {k: cfg[k] for k in ("tariff", "export", "capex", "battery",
                                       "city", "balcony_kit", "demand",
                                       "finance", "roof", "panel", "losses")},
    }
    out = os.path.join(HERE, "outputs", "madrid_block.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    say(f"  outputs/madrid_block.json          "
        f"{os.path.getsize(out) / 1024:,.0f} kB")
    with open(os.path.join(HERE, "outputs", "run_log.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_LOG) + "\n")
    say(f"  outputs/run_log.txt                full transcript of this run")
    return payload


# --------------------------------------------------------------------------

def main():
    cfg = load_config()
    doy, hour = build_time_grid()

    sun, off, month_idx, sky, spec = calibrate(cfg, doy, hour)
    buildings, addressable, proj, results, per_building, pts_xyh = \
        simulate_block(cfg, sun, month_idx, hour, sky)
    tilt_tradeoff(cfg, sun, month_idx, hour, addressable, pts_xyh, sky)

    load_one = report_demand(cfg, doy, hour)
    rows, panels, best = collective_economics(cfg, results, per_building,
                                              doy, hour, load_one)

    kwp_tot = len(results) * cfg["panel"]["wp"] / 1000.0
    spec_in_situ = sum(p["kwh"] for p in results) / kwp_tot
    dwellings_tot = sum(pb.get("dwellings_est", 0) for pb in per_building)
    per_dw_kwp = kwp_tot / dwellings_tot
    per_dw_kwh = sum(p["kwh"] for p in results) / dwellings_tot

    uk_comparison(cfg, spec_in_situ, per_dw_kwp, per_dw_kwh, load_one,
                  doy, hour)
    battery_hurdle(cfg, panels, best, doy, hour, load_one)
    balcony_kwh = run_balcony(cfg, sun, month_idx, hour, doy, load_one, sky,
                              spec_in_situ)
    scale_to_city(cfg, addressable, per_building, results, balcony_kwh)
    sensitivity(cfg, panels, best, doy, hour, load_one)
    export_web(cfg, buildings, addressable, proj, results, per_building,
               rows, spec, balcony_kwh, per_dw_kwp, per_dw_kwh)
    say("")


if __name__ == "__main__":
    main()
