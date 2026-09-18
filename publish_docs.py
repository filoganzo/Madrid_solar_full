#!/usr/bin/env python3
"""
publish_docs.py — publish the single-page city map into docs/ for GitHub Pages.

Pages serves the repository root or /docs and nothing else, while the build
scripts write into web/. Rather than restructure the project around a hosting
quirk, this copies what should be public into docs/ and leaves web/ as the
build output.

    python publish_docs.py

Writes:
    docs/index.html          the city map
    docs/tiles/*.json        per-panel detail, fetched on demand
    docs/.nojekyll           stops Pages running Jekyll over the files
"""

from __future__ import annotations

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")
DOCS = os.path.join(HERE, "docs")


def copy(src: str, dst: str) -> int:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return os.path.getsize(dst)


def main() -> int:
    city_src = os.path.join(WEB, "city")
    city_idx = os.path.join(city_src, "index.html")
    if not os.path.exists(city_idx):
        print("web/city/index.html missing - run `python build_city_web.py` first",
              file=sys.stderr)
        return 1

    if os.path.isdir(DOCS):
        shutil.rmtree(DOCS)
    os.makedirs(DOCS, exist_ok=True)

    # .nojekyll: without it Pages runs Jekyll, which ignores files and
    # directories beginning with an underscore and is pure downside here.
    with open(os.path.join(DOCS, ".nojekyll"), "w", encoding="utf-8") as fh:
        fh.write("")

    total = 0
    total += copy(city_idx, os.path.join(DOCS, "index.html"))
    tiles_src = os.path.join(city_src, "tiles")
    n = 0
    if os.path.isdir(tiles_src):
        for name in os.listdir(tiles_src):
            total += copy(os.path.join(tiles_src, name),
                          os.path.join(DOCS, "tiles", name))
            n += 1
    print(f"  docs/index.html  the city map")
    print(f"  docs/tiles/       {n} tiles")

    print()
    print(f"  {total / 1e6:.1f} MB in docs/")
    print("  commit docs/ and set Pages to deploy from main /docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
