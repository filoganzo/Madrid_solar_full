#!/usr/bin/env python3
"""
analyse_city.py — what the city run says that the block run could not.

Section 11 of the block study extrapolates: it takes 0.97 kWp per dwelling,
measured on one Ensanche block, multiplies it by an estimate of Madrid's
multifamily dwellings, and prints a range. That line carries the study's
loudest health warning — `conf: LOW, most sensitive input in this section` —
because a single block of 1860-1930 mansion blocks is not a city, and the
whole extrapolation rests on it.

The city run exists to replace that multiplication with a measurement. This
script does the comparison honestly, which means reporting the places where
the extrapolation was wrong as prominently as the places it held.

    python analyse_city.py

Reads web/city/city.json; writes outputs/city_findings.txt.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# Windows consoles default to a legacy codepage, which turns the rule
# characters below into '?'. The file on disk was always correct UTF-8; this
# makes the terminal agree with it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_LOG: list[str] = []


def say(line: str = "") -> None:
    print(line)
    _LOG.append(line)


def rule(title: str) -> None:
    say("")
    say("=" * 76)
    say(title)
    say("=" * 76)


def main() -> int:
    city_path = os.path.join(HERE, "web", "city", "city.json")
    if not os.path.exists(city_path):
        print("no web/city/city.json - run `python run_city.py` first",
              file=sys.stderr)
        return 1

    with open(city_path, encoding="utf-8") as fh:
        city = json.load(fh)
    with open(os.path.join(HERE, "config", "madrid.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    meta = city["meta"]
    blds = city["buildings"]
    modelled = [b for b in blds if b["n"] > 0]
    if not modelled:
        print("no modelled roofs in city.json", file=sys.stderr)
        return 1

    wp = meta["panel_wp"]
    kwp = np.array([b["n"] * wp / 1000.0 for b in modelled])
    kwh = np.array([b["k"] for b in modelled], dtype=float)
    dwell = np.array([b["d"] for b in modelled], dtype=float)
    area = np.array([b["a"] for b in modelled], dtype=float)
    per_panel = kwh / np.array([b["n"] for b in modelled])

    rule("1 · THE EXTRACT")
    say(f"  buildings drawn                    {meta['buildings']:,}")
    say(f"  roofs modelled                     {len(modelled):,}")
    say(f"  panels                             {meta['panels']:,}")
    say(f"  capacity                           {kwp.sum() / 1000:,.1f} MWp")
    say(f"  generation                         {kwh.sum() / 1e6:,.1f} GWh/yr")
    say(f"  simulation time                    "
        f"{meta['sim_seconds'] / 60:,.1f} min on "
        f"{meta['panels'] / max(meta['sim_seconds'], 1):,.0f} panels/s")
    say("")
    say("  Every building in the extract shades, whether or not it carries")
    say("  panels. That is why the unmodelled ones are still loaded.")

    # ---------------------------------------------------------------- roof
    rule("2 · ROOF PER DWELLING — the headline result, now measured")
    total_kwp, total_dw = kwp.sum(), dwell.sum()
    city_ratio = total_kwp / total_dw
    per_bld = kwp / np.maximum(dwell, 1)
    block_ratio = 0.97          # the Ensanche block, from run_madrid.py

    say(f"  dwellings under a modelled roof    {total_dw:,.0f}")
    say(f"  capacity over those dwellings      {total_kwp:,.0f} kWp")
    say(f"  ROOF PER DWELLING, aggregate       {city_ratio:.2f} kWp"
        f"   ({city_ratio * float(np.median(per_panel)) / (wp / 1000):,.0f} kWh/yr)")
    say("")
    say(f"  the single Ensanche block said     {block_ratio:.2f} kWp")
    say(f"  difference                         "
        f"{(city_ratio / block_ratio - 1) * 100:+.0f}%")
    say("")
    say("  Distribution across buildings, which the block could not show:")
    for p in (5, 25, 50, 75, 95):
        say(f"    {p:2d}th percentile                 "
            f"{np.percentile(per_bld, p):.2f} kWp/dwelling")
    say("")
    share_below_1 = float((per_bld < 1.0).mean()) * 100
    say(f"  buildings under 1 kWp/dwelling     {share_below_1:.0f}%")
    say("  A single block cannot produce a distribution, and the spread here")
    say("  is the point: the mean is not the building you are selling to.")

    # ---------------------------------------------------------------- yield
    rule("3 · YIELD — does the block's number survive the city?")
    say(f"  per-panel yield, median            "
        f"{np.median(per_panel):,.0f} kWh/yr")
    say(f"  5th / 95th percentile              "
        f"{np.percentile(per_panel, 5):,.0f} / "
        f"{np.percentile(per_panel, 95):,.0f} kWh/yr")
    say(f"  specific yield                     "
        f"{kwh.sum() / kwp.sum():,.0f} kWh/kWp")
    say(f"  PVGIS unshaded at 37 deg           "
        f"{cfg['calibration']['pvgis_specific_yield_kwh_kwp']:,} kWh/kWp")
    gap = (1 - (kwh.sum() / kwp.sum())
           / cfg["calibration"]["pvgis_specific_yield_kwh_kwp"]) * 100
    say(f"  cost of the city, in yield         -{gap:.0f}%")
    say("")
    say("  Read that against finding 7 of the block study: most of this gap is")
    say("  the deliberate 15 deg tilt, not the neighbours. The city run does")
    say("  not overturn that; it widens the sample it rests on.")

    # ------------------------------------------------------------ provenance
    rule("4 · PROVENANCE — how much of this is measured")
    src = np.array([b["s"] for b in blds])
    names = {0: "inferred from typology", 1: "OSM floor count", 2: "OSM height"}
    for k in (0, 1, 2):
        n = int((src == k).sum())
        say(f"  {names[k]:34s} {n:7,d}  ({n / len(blds) * 100:4.1f}%)")
    measured = float((src > 0).mean()) * 100
    say("")
    say(f"  measured, not inferred             {measured:.1f}%")
    say("  This is the same weakness the block study named, at city scale and")
    say("  therefore harder to wave away. It is third on the list to fix, not")
    say("  first, for the reason attribute_shading.py established: urban")
    say("  obstruction is worth ~2% of yield in a city built to a cornice line.")

    # ------------------------------------------------------------ the point
    rule("5 · WHAT THIS CHANGES")
    dwell_multi = cfg["city"]["dwellings_total"] * cfg["city"]["multifamily_share"] \
        if "dwellings_total" in cfg.get("city", {}) else 1_144_000
    say("  Section 11 of the block study extrapolates one block to Madrid and")
    say("  flags the ratio it uses as the most sensitive input in the section.")
    say("  This run replaces that assumption with a measurement over"
        f" {len(modelled):,}")
    say("  roofs, and the ratio moves:")
    say("")
    say(f"    block-derived      0.97 kWp/dwelling  ->  "
        f"{dwell_multi * 0.97 / 1e6:,.2f} GW")
    say(f"    city-measured      {city_ratio:.2f} kWp/dwelling  ->  "
        f"{dwell_multi * city_ratio / 1e6:,.2f} GW")
    say("")
    say("  The honest caveat, stated as plainly as the number: this extract is")
    say("  still central Madrid, so it is a wider sample of the same kind of")
    say("  city rather than a survey of every typology in it. Vallecas and the")
    say("  post-war periphery have sloped, lower, differently-owned roofs, and")
    say("  they are still under-represented here.")
    say("")
    say("  What does not move is the constraint. A larger roof per dwelling")
    say("  changes the size of the prize; it does not change who has to sign.")
    say("  The binding input is still a junta de propietarios vote, and the")
    say("  deployment rate is still set by the sales cycle rather than the sun.")

    out = os.path.join(HERE, "outputs", "city_findings.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print()
    print(f"  written to {os.path.relpath(out, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
