#!/usr/bin/env python3
"""
run_city.py — the block model, run over Madrid.

The physics here is not new. Every panel goes through exactly the same chain
as in run_madrid.py: NOAA solar position, Ineichen-Perez clear sky, day-type
clearness, Erbs split, Hay-Davies transposition onto a 15 deg plane, a
geometric horizon from real neighbouring footprints, Faiman cell temperature
and the same loss chain. What changes is only what it takes to run that chain
about two hundred times more often, and what it takes to put the answer in a
browser.

Three things had to change, and each is a consequence of scale rather than a
new modelling idea:

  1  Neighbours are found through a uniform grid (src/spatial.py) instead of
     by scanning every densified outline point. build_horizon_fast already
     discards everything past 400 m, so restricting the scan to the cells
     inside that radius returns the same points and the same horizon exactly.
     tests/test_city.py asserts that equality rather than assuming it.

  2  Buildings are simulated in parallel. Each building's yield depends only
     on its neighbours' geometry, never on another building's result, so the
     work is embarrassingly parallel and the only shared state is read-only.

  3  Output is tiled. The honest constraint at city scale is not compute, it
     is the browser: every panel in Madrid as one JSON is roughly a gigabyte.
     So the city view ships one summary row per building, and per-panel
     detail goes to small per-tile files the map fetches only for the area
     actually on screen.

    python run_city.py                    # the fetched AOI
    python run_city.py --limit 2000       # a quick subset while developing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import demand, irradiance, panel, roof_model, shading
from src.spatial import PointGrid
from src.solar_geometry import is_summer_time, sun_position

HERE = os.path.dirname(os.path.abspath(__file__))

# Windows consoles default to a legacy codepage, which turns the rule
# characters below into '?'. The file on disk was always correct UTF-8; this
# makes the terminal agree with it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

YEAR = 2026
TILE_DEG = 0.005                    # ~550 m: a few blocks per tile
SHADE_RADIUS_M = 400.0

# Worker globals: built once per process, never mutated afterwards.
_W: dict = {}


# --------------------------------------------------------------------------
# Shared sky, built once and reused by every worker
# --------------------------------------------------------------------------

def build_sky(cfg: dict):
    """The hourly sun and sky for the year. Identical to run_madrid.py."""
    doy = np.repeat(np.arange(1, 366), 24)
    hour = np.tile(np.arange(24) + 0.5, 365)
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
    return sun, month_idx, hour, (ghi, dni, dhi)


def _init_worker(cfg, pts_xyh, cell_m):
    """Runs once per process. Rebuilds the sky locally so the big hourly
    arrays are never pickled per task."""
    sun, month_idx, hour, sky = build_sky(cfg)
    _W.update(cfg=cfg, sun=sun, month_idx=month_idx, hour=hour, sky=sky,
              grid=PointGrid(pts_xyh, cell_m))


def _simulate_building(payload):
    """One building: pack panels, shade each, return per-panel yields."""
    cfg = _W["cfg"]
    sun = _W["sun"]
    grid = _W["grid"]
    month_idx = _W["month_idx"]
    hour = _W["hour"]
    ghi, dni, dhi = _W["sky"]
    r = cfg["roof"]

    (osm_id, poly, height_m, height_source, levels, area_m2,
     use, flat_roof, tags) = payload
    b = roof_model.Building(
        osm_id=osm_id, polygon=np.asarray(poly), height_m=height_m,
        height_source=height_source, levels=levels, use=use, tags=tags,
        area_m2=area_m2, flat_roof=flat_roof)

    slots, _ = roof_model.pack_panels(b, cfg)
    if not slots:
        return osm_id, []

    up_slope = cfg["panel"]["width_m"] * np.cos(np.radians(r["tilt_deg"]))
    pitch = up_slope * r["row_pitch_factor"]

    # One neighbourhood query per building rather than per panel: every panel
    # on a roof shares the same 400 m surroundings to well within a cell.
    cx = float(np.mean(b.polygon[:, 0]))
    cy = float(np.mean(b.polygon[:, 1]))
    span = float(np.hypot(np.ptp(b.polygon[:, 0]), np.ptp(b.polygon[:, 1])))
    local = grid.query(cx, cy, SHADE_RADIUS_M + span)

    out = []
    for s in slots:
        hz = shading.build_horizon_fast(np.array([s.x, s.y, s.z]), local)
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
        out.append((s.x, s.y, s.z, float(np.nansum(kwh)), float(svf)))
    return osm_id, out


# --------------------------------------------------------------------------

def estimate_dwellings(area_m2: float, levels: int) -> int:
    """Same rule as run_madrid.py: net floor area over a mean flat size."""
    net = area_m2 * max(levels - 1, 1) * 0.75
    return max(int(round(net / 90.0)), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.path.join(
        HERE, "data", "city", "osm_city.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "web", "city"))
    ap.add_argument("--limit", type=int, default=0,
                    help="simulate only the N largest roofs (development)")
    ap.add_argument("--workers", type=int,
                    default=max((os.cpu_count() or 4) - 2, 1))
    args = ap.parse_args()

    t_start = time.time()
    cfg_path = os.path.join(HERE, "config", "madrid.yaml")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    print("loading " + os.path.basename(args.source) + " ...", flush=True)
    buildings, proj = roof_model.load_osm(args.source, cfg)
    print(f"  buildings parsed                 {len(buildings):,}")

    # Every building shades, whether or not it can carry panels.
    obstacles = roof_model.to_obstacles(buildings)
    pts = shading.densify_all(obstacles)
    print(f"  densified outline points         {len(pts):,}")

    addressable = [b for b in buildings if b.is_addressable]
    if args.limit:
        addressable = sorted(addressable, key=lambda b: -b.area_m2)[:args.limit]
    print(f"  addressable roofs                {len(addressable):,}")

    payloads = [(b.osm_id, b.polygon.tolist(), b.height_m, b.height_source,
                 b.levels, b.area_m2, b.use, b.flat_roof, b.tags)
                for b in addressable]

    print(f"  simulating on {args.workers} processes ...", flush=True)
    t0 = time.time()
    results: dict[int, list] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(cfg, pts, 100.0)) as pool:
        for osm_id, panels_out in pool.map(_simulate_building, payloads,
                                           chunksize=8):
            results[osm_id] = panels_out
            done += 1
            if done % 250 == 0 or done == len(payloads):
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                eta = (len(payloads) - done) / max(rate, 1e-9) / 60.0
                print(f"    {done:,}/{len(payloads):,} roofs   "
                      f"{rate:5.1f}/s   eta {eta:5.1f} min", flush=True)
    sim_s = time.time() - t0

    # ---------------------------------------------------------------- export
    os.makedirs(args.out, exist_ok=True)
    tiles_dir = os.path.join(args.out, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    for stale in os.listdir(tiles_dir):
        os.remove(os.path.join(tiles_dir, stale))

    tiles: dict[str, list] = {}
    city_rows = []
    total_panels, total_kwh = 0, 0.0
    all_lat, all_lon = [], []

    for b in buildings:
        lat, lon = proj.to_latlon(b.polygon[:, 0], b.polygon[:, 1])
        lat = np.atleast_1d(lat)
        lon = np.atleast_1d(lon)
        all_lat.append(lat)
        all_lon.append(lon)
        ring = [[round(float(la), 6), round(float(lo), 6)]
                for la, lo in zip(lat, lon)]
        pans = results.get(b.osm_id, [])
        kwh = sum(p[3] for p in pans)
        row = {
            "i": b.osm_id,
            "g": ring,
            "h": round(b.height_m, 1),
            "lv": b.levels,
            "s": {"osm:height": 2, "osm:levels": 1}.get(b.height_source, 0),
            "u": {"residential": 0, "office": 1}.get(b.use, 2),
            "a": round(b.area_m2),
            "n": len(pans),
            "k": round(kwh),
            "d": estimate_dwellings(b.area_m2, b.levels),
        }
        city_rows.append(row)
        total_panels += len(pans)
        total_kwh += kwh

        if pans:
            clat = float(np.mean(lat))
            clon = float(np.mean(lon))
            key = f"{math.floor(clat / TILE_DEG)}_{math.floor(clon / TILE_DEG)}"
            plat, plon = proj.to_latlon(
                np.array([p[0] for p in pans]),
                np.array([p[1] for p in pans]))
            tiles.setdefault(key, []).append({
                "i": b.osm_id,
                "p": [[round(float(la), 6), round(float(lo), 6),
                       round(p[3]), round(p[4], 2)]
                      for la, lo, p in zip(np.atleast_1d(plat),
                                           np.atleast_1d(plon), pans)],
            })

    for key, payload in tiles.items():
        with open(os.path.join(tiles_dir, key + ".json"), "w",
                  encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))

    lat_all = np.concatenate(all_lat)
    lon_all = np.concatenate(all_lon)
    city = {
        "meta": {
            "generated": time.strftime("%Y-%m-%d %H:%M"),
            "buildings": len(buildings),
            "simulated": len(addressable),
            "panels": total_panels,
            "kwh": round(total_kwh),
            "kwp": round(total_panels * cfg["panel"]["wp"] / 1000.0),
            "tile_deg": TILE_DEG,
            "sim_seconds": round(sim_s),
            "bbox": [round(float(lat_all.min()), 5),
                     round(float(lon_all.min()), 5),
                     round(float(lat_all.max()), 5),
                     round(float(lon_all.max()), 5)],
            "panel_wp": cfg["panel"]["wp"],
        },
        "buildings": city_rows,
    }
    city_path = os.path.join(args.out, "city.json")
    with open(city_path, "w", encoding="utf-8") as fh:
        json.dump(city, fh, separators=(",", ":"))

    mb = os.path.getsize(city_path) / 1e6
    tile_mb = sum(os.path.getsize(os.path.join(tiles_dir, f))
                  for f in os.listdir(tiles_dir)) / 1e6
    print()
    print(f"  panels simulated                 {total_panels:,}")
    print(f"  installed capacity               "
          f"{total_panels * cfg['panel']['wp'] / 1e6:,.1f} MWp")
    print(f"  annual generation                {total_kwh / 1e6:,.1f} GWh")
    print(f"  simulation time                  {sim_s / 60:.1f} min "
          f"({total_panels / max(sim_s, 1e-9):,.0f} panels/s)")
    print(f"  city.json                        {mb:.1f} MB")
    print(f"  {len(tiles)} panel tiles                    {tile_mb:.1f} MB")
    print(f"  total wall clock                 "
          f"{(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
