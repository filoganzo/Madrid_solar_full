#!/usr/bin/env python3
"""
build_web.py — assemble web/index.html as one self-contained file.

    python run_madrid.py && python build_web.py && python tests/check_js_port.py

The page is a single HTML file with the data inlined, on purpose: it drops
onto any static host with no build step, no API key and no tile server, and
it keeps working offline. The block is drawn from real footprints as vector
geometry rather than over a basemap, which also means nothing here depends
on a third party staying up.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

KEEP_CFG = ("tariff", "export", "city", "balcony_kit", "demand", "roof",
            "panel")
SRC_CODE = {"osm:height": 2, "osm:levels": 1, "inferred": 0}
USE_CODE = {"residential": 0, "office": 1, "other": 2}


def slim(full: dict) -> dict:
    """Shrink the run output for the browser.

    Field names are single letters and everything is rounded. Not
    micro-optimisation: the payload is inlined into the page, so its size is
    the page's load time, and rounding a yield figure to the nearest kWh
    costs nothing a reader could use.
    """
    return {
        "meta": full["meta"],
        "buildings": [{
            "i": b["id"],
            "p": [[round(x, 1), round(y, 1)] for x, y in b["xy"]],
            "h": b["h"], "lv": b["lv"], "s": SRC_CODE[b["src"]],
            "u": USE_CODE[b["use"]], "a": b["area"],
            "ad": 1 if b["addr"] else 0,
            "n": b["name"][:40], "st": b["street"][:40],
        } for b in full["buildings"]],
        "panels": [{
            "b": p["b"], "x": p["xy"][0], "y": p["xy"][1], "z": p["z"],
            "k": p["kwh"], "v": round(p["svf"], 2), "m": p["m"],
        } for p in full["panels"]],
        "pb": {str(x["osm_id"]): {
            "n": x.get("panels", 0), "kwp": round(x.get("kwp", 0), 1),
            "kwh": round(x.get("kwh", 0)), "dw": x.get("dwellings_est", 0),
            "lv": x.get("levels", 0), "a": round(x.get("area_m2", 0)),
        } for x in full["per_building"]},
        "econ": full["economics"],
        "balcony": full["balcony_kwh"],
        "cfg": {k: full["config"][k] for k in KEEP_CFG},
    }


def main() -> None:
    src = os.path.join(HERE, "outputs", "madrid_block.json")
    if not os.path.exists(src):
        raise SystemExit("run `python run_madrid.py` first")
    payload = slim(json.load(open(src, encoding="utf-8")))
    data = json.dumps(payload, separators=(",", ":"))
    with open(os.path.join(HERE, "web", "data.json"), "w",
              encoding="utf-8") as f:
        f.write(data)

    tpl = open(os.path.join(HERE, "web", "template.html"),
               encoding="utf-8").read()
    if "__DATA__" not in tpl:
        raise SystemExit("template.html has no __DATA__ placeholder")
    out = os.path.join(HERE, "web", "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(tpl.replace("__DATA__", data))
    print(f"web/index.html   {os.path.getsize(out) / 1024:,.0f} kB"
          f"   ({len(payload['buildings'])} buildings, "
          f"{len(payload['panels']):,} panels)")
    print("next: python tests/check_js_port.py")


if __name__ == "__main__":
    main()
