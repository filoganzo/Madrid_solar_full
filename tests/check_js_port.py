#!/usr/bin/env python3
"""
check_js_port.py — prove the map is telling the truth.

    python tests/check_js_port.py      (needs node on PATH)

web/index.html contains a JavaScript port of src/solar_geometry.py, because
the map has to move the sun in the browser. Two implementations of the same
equations is a standing invitation for them to drift apart, and a drifted
port is worse than no map at all: it would draw shadows that the economics
never saw.

So this script extracts the JavaScript straight out of the built page, runs
it under node, and compares it against the Python model at six awkward
moments — near sunrise, near sunset, both solstices, an equinox, and a
December afternoon. Awkward moments on purpose: that is where refraction and
the azimuth quadrant logic are most likely to disagree, and a midday-only
check would pass while both were wrong.

It is deliberately not a pytest case, because it needs node and the built
HTML rather than just the package. It is run before publishing the page.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.solar_geometry import is_summer_time, sun_position  # noqa: E402

LAT, LON = 40.4313, -3.6883
TOLERANCE_DEG = 0.01

CASES = [
    (80, 13.50, "equinox, solar noon"),
    (172, 14.25, "June solstice, solar noon"),
    (355, 13.25, "December solstice, solar noon"),
    (172, 8.00, "June, early morning"),
    (172, 20.00, "June, late evening"),
    (355, 16.50, "December, late afternoon"),
]

JS_DRIVER = """
const fs = require('fs');
const html = fs.readFileSync(%(page)s, 'utf8');
const start = html.indexOf('const DIM =');
const end = html.indexOf('/* ===================== bounds');
if (start < 0 || end < 0) { console.error('could not locate the port'); process.exit(2); }
const META = {lat: %(lat)s, lon: %(lon)s, utc_winter: 1, utc_summer: 2};
eval(html.slice(start, end));
const cases = %(cases)s;
console.log(JSON.stringify(cases.map(c => {
  const s = sunPos(META.lat, META.lon, c[0], c[1]);
  return [s.alt, s.az];
})));
"""


def main() -> int:
    page = os.path.join(ROOT, "web", "index.html")
    if not os.path.exists(page):
        print("web/index.html not built yet — run build_web.py first")
        return 2

    driver = JS_DRIVER % {
        "page": json.dumps(page),
        "lat": LAT, "lon": LON,
        "cases": json.dumps([[d, h] for d, h, _ in CASES]),
    }
    try:
        out = subprocess.run(["node", "-e", driver], capture_output=True,
                             text=True, timeout=60, check=True)
    except FileNotFoundError:
        print("node not found on PATH — skipping the port check")
        return 0
    except subprocess.CalledProcessError as exc:
        print("node failed:\n" + exc.stderr)
        return 1

    js = json.loads(out.stdout.strip().splitlines()[-1])

    print(f"{'case':30s} {'alt py':>8} {'alt js':>8} {'d':>8}"
          f" {'az py':>10} {'az js':>10} {'d':>8}")
    worst = 0.0
    for (doy, hour, label), (ja, jz) in zip(CASES, js):
        off = 2 if is_summer_time(doy) else 1
        s = sun_position(LAT, LON, 2026, np.array([doy]), np.array([hour]),
                         off)
        pa, pz = float(s.altitude_deg[0]), float(s.azimuth_deg[0])
        worst = max(worst, abs(pa - ja), abs(pz - jz))
        print(f"{label:30s} {pa:8.3f} {ja:8.3f} {pa - ja:+8.4f}"
              f" {pz:10.3f} {jz:10.3f} {pz - jz:+8.4f}")

    print(f"\nworst divergence: {worst:.5f} deg   "
          f"tolerance: {TOLERANCE_DEG} deg")
    if worst > TOLERANCE_DEG:
        print("FAIL — the map and the model have drifted apart.")
        return 1
    print("PASS — the map and the model agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
