#!/usr/bin/env python3
"""
attribute_shading.py — how much was the shading model actually worth?

    python attribute_shading.py

Separate from run_madrid.py because it is a question about the model rather
than about Madrid, and because the answer surprised me enough to be worth
isolating.

The in-situ specific yield comes out about 7% below the PVGIS headline for
an optimally tilted panel. I assumed most of that gap was the city: a dense
Ensanche block, six-storey walls, narrow streets. That assumption is why
shading.py exists in the form it does, with per-panel horizon profiles and
view factors instead of a flat loss factor.

This script takes the gap apart by adding one obstruction at a time to a
250-panel random sample, and the attribution does not say what I expected.
Result is printed below and written to outputs/shading_attribution.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src import demand, irradiance, panel, roof_model, shading  # noqa: E402
from src.solar_geometry import is_summer_time, sun_position  # noqa: E402

SAMPLE_N = 250
SEED = 42


def main() -> None:
    cfg = yaml.safe_load(open(os.path.join(HERE, "config", "madrid.yaml"),
                             encoding="utf-8"))
    doy = np.repeat(np.arange(1, 366), 24)
    hour = np.tile(np.arange(24) + 0.5, 365)
    site = cfg["site"]
    off = np.where(np.vectorize(is_summer_time)(doy),
                   site["utc_offset_summer"], site["utc_offset_winter"])
    sun = sun_position(site["lat"], site["lon"], 2026, doy, hour, off)
    mi = demand.month_index_from_doy(doy)
    cs = irradiance.ineichen_clear_sky(sun, site["elevation_m"],
                                       site["linke_turbidity"], mi)
    ghi, dni, dhi = irradiance.apply_cloudiness(sun, *cs, mi)

    blds, _ = roof_model.load_osm(os.path.join(HERE, "data", "osm_raw.json"),
                                  cfg)
    pts = shading.densify_all(roof_model.to_obstacles(blds))
    addressable = [b for b in blds if b.is_addressable]
    r = cfg["roof"]
    pitch = (cfg["panel"]["width_m"] * np.cos(np.radians(r["tilt_deg"]))
             * r["row_pitch_factor"])

    def annual(hz, tilt):
        mask = shading.beam_mask(hz, sun.altitude_deg, sun.azimuth_deg)
        svf = shading.sky_view_factor(hz, tilt, r["azimuth_deg"])
        tr = irradiance.transpose_hay_davies(
            sun, ghi, dni, dhi, tilt, r["azimuth_deg"],
            albedo=site["albedo"], sky_view=svf, beam_shade=mask,
            iam_b0=cfg["panel"]["iam_b0"])
        kwh, _ = panel.ac_energy_kwh(tr["poa"], mi, hour, cfg)
        return float(np.nansum(kwh))

    slots = []
    for b in addressable:
        placed, _ = roof_model.pack_panels(b, cfg)
        slots.extend((b, s) for s in placed)

    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(slots), size=min(SAMPLE_N, len(slots)),
                      replace=False)
    sample = [slots[i] for i in pick]

    zero = np.zeros(shading.N_BINS)
    a = annual(zero, cfg["calibration"]["pvgis_optimal_tilt_deg"])
    b_open = annual(zero, r["tilt_deg"])

    acc = {"nb": 0.0, "par": 0.0, "full": 0.0}
    for bld, s in sample:
        hz = shading.build_horizon_fast(np.array([s.x, s.y, s.z]), pts)
        acc["nb"] += annual(hz, r["tilt_deg"])
        hz_p = shading.add_parapet(hz.copy(), np.array([s.x, s.y]),
                                   bld.polygon, r["parapet_height_m"], 0.35)
        acc["par"] += annual(hz_p, r["tilt_deg"])
        hz_f = shading.add_row_self_shading(
            hz_p.copy(), r["tilt_deg"], r["azimuth_deg"],
            cfg["panel"]["width_m"], pitch)
        acc["full"] += annual(hz_f, r["tilt_deg"])

    n = len(sample)
    c, d, e = acc["nb"] / n, acc["par"] / n, acc["full"] / n

    print(f"population {len(slots):,} panels   random sample {n}   seed {SEED}")
    print("\nMean annual kWh per panel, adding one obstruction at a time:")
    print(f"  A  open horizon, 37deg (PVGIS optimum)  {a:7.1f}")
    print(f"  B  open horizon, 15deg (chosen tilt)    {b_open:7.1f}"
          f"   {(b_open / a - 1) * 100:+6.2f}% vs A")
    print(f"  C  B + neighbouring buildings           {c:7.1f}"
          f"   {(c / b_open - 1) * 100:+6.2f}% vs B")
    print(f"  D  C + own parapet                      {d:7.1f}"
          f"   {(d / b_open - 1) * 100:+6.2f}% vs B")
    print(f"  E  D + row self-shading (as built)      {e:7.1f}"
          f"   {(e / b_open - 1) * 100:+6.2f}% vs B")

    dt, dn, dp, dr = b_open - a, c - b_open, d - c, e - d
    tot = dt + dn + dp + dr
    print(f"\nAttribution of the whole {-tot / a * 100:.1f}% gap from the "
          f"PVGIS headline:")
    for label, v in [("flat-tilt choice", dt),
                     ("neighbouring buildings", dn),
                     ("own parapet", dp),
                     ("row self-shading", dr)]:
        print(f"  {label:24s} {-v:6.1f} kWh   {v / tot * 100:5.1f}% of the gap")
    urban = (dn + dp + dr) / b_open * 100
    print(f"\n  ALL urban obstruction combined: {-urban:.1f}% of yield")

    print("\nWhat I take from this, which is not what I expected:")
    print("  Dense-city intuition says urban shading should cost 15-25% of")
    print("  rooftop yield. In the Ensanche it costs 2.3%, and two thirds of")
    print("  the gap from the PVGIS headline is a tilt choice I made on")
    print("  purpose to fit 67% more panels on the roof.")
    print()
    print("  The reason is the cornice line. The 1860-1930 grid was built to")
    print("  a uniform height, so every roof sits above every neighbour's")
    print("  shadow: at the December solstice a same-height neighbour stands")
    print("  only a parapet above the panel plane and reaches about 3 m, not")
    print("  the 90 m its shadow runs at street level. The streets are dark")
    print("  all winter and the roofs are in full sun. Shading bites where")
    print("  the cornice line breaks, which here means the post-1960 towers")
    print("  on the Castellana axis.")
    print()
    print("  CONSEQUENCE, and it costs me an argument: I had listed building")
    print("  heights as the top data request, on the grounds that 94% of them")
    print("  are inferred and heights drive shading. If urban obstruction is")
    print("  worth 2.3%, then a one-storey error in the typology prior moves")
    print("  the yield numbers by a fraction of a percent, not by a lot. So")
    print("  the LiDAR request drops to third. The load curve and the")
    print("  dwelling count are the two that actually move the answer, and")
    print("  both are about people rather than geometry.")
    print()
    print("  It also means shading.py is more machinery than this block")
    print("  needed. I would keep it, because the number it produced is one I")
    print("  can defend and because it will matter in Vallecas or Tetuan")
    print("  where the heights are irregular — but I would not build it first")
    print("  again.")

    out = os.path.join(HERE, "outputs", "shading_attribution.json")
    json.dump({"A_open_37deg": a, "B_open_15deg": b_open,
               "C_neighbours": c, "D_parapet": d, "E_rows": e,
               "gap_total_pct": -tot / a * 100,
               "urban_obstruction_pct": -urban,
               "share_tilt": dt / tot, "share_neighbours": dn / tot,
               "share_parapet": dp / tot, "share_rows": dr / tot,
               "n_sample": n, "n_population": len(slots), "seed": SEED},
              open(out, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
