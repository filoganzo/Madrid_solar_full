#!/usr/bin/env python3
"""
build_city_web.py — assemble the city map from the template and the run.

The block map (build_web.py) inlines its data, because 600 kB in one file is
a page you can email. The city map cannot: the panel layer is tens of
megabytes and the browser should only ever hold the part you are looking at.
So this writes an index.html with the building summary inlined — that layer
is always needed, and inlining it removes a round trip and a CORS problem —
and leaves the per-tile panel files on disk beside it.

One consequence worth stating plainly: because the tiles are fetched, the
city page needs to be served over http rather than opened from file://.
`python -m http.server` is enough, and that is in the README.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CITY_DIR = os.path.join(HERE, "web", "city")


def main() -> int:
    city_json = os.path.join(CITY_DIR, "city.json")
    if not os.path.exists(city_json):
        print("no web/city/city.json — run `python run_city.py` first",
              file=sys.stderr)
        return 1

    with open(os.path.join(HERE, "web", "city_template.html"),
              encoding="utf-8") as fh:
        tpl = fh.read()
    with open(city_json, encoding="utf-8") as fh:
        city = json.load(fh)

    html = (tpl
            .replace("__CITY_JSON__", json.dumps(city, separators=(",", ":")))
            .replace("__TILE_BASE__", json.dumps("tiles/")))

    out = os.path.join(CITY_DIR, "index.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)

    tiles_dir = os.path.join(CITY_DIR, "tiles")
    n_tiles = len(os.listdir(tiles_dir)) if os.path.isdir(tiles_dir) else 0
    tile_mb = 0.0
    if n_tiles:
        tile_mb = sum(os.path.getsize(os.path.join(tiles_dir, f))
                      for f in os.listdir(tiles_dir)) / 1e6

    m = city["meta"]
    print(f"  buildings inlined   {m['buildings']:,}")
    print(f"  panels in tiles     {m['panels']:,}  ({n_tiles} tiles, "
          f"{tile_mb:.1f} MB)")
    print(f"  {out}   {os.path.getsize(out) / 1e6:.1f} MB")
    print()
    print("  serve it:  python -m http.server -d web/city 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
