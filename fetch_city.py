#!/usr/bin/env python3
"""
fetch_city.py — pull Madrid's building footprints from Overpass, tile by tile.

Why tiles: a single city-wide Overpass query for ~125,000 buildings times out
(it did, twice, with a 504). The city is therefore cut into a grid of small
bboxes, each fetched and cached separately. A tile that fails is retried with
backoff and, failing that, split in four. The cache means an interrupted run
resumes instead of restarting — which matters when the whole fetch is the
slowest part of the pipeline and the endpoint is a shared free service.

Politeness is deliberate: one request at a time, a pause between tiles, a
descriptive User-Agent. Overpass is donated infrastructure.

    python fetch_city.py                 # the default AOI
    python fetch_city.py --all-madrid    # the full municipality
"""
import argparse, json, math, os, re, sys, time, urllib.error, urllib.parse
import urllib.request

UA = "madrid-solar-study/1.0 (rooftop PV case study)"

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "city", "tiles")

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
]
STATUS_URL = "https://overpass-api.de/api/status"

# Overpass publishes its own rate limit and says when the next slot frees.
# Reading that is the difference between a fetch that finishes and one that
# collects 429s until it gives up: the server is not being unfriendly, it is
# telling you exactly how fast you may go, and the first two attempts at this
# script both died because they did not listen.


def wait_for_slot(max_wait: float = 180.0) -> None:
    """Block until the endpoint reports a free slot."""
    try:
        req = urllib.request.Request(
            STATUS_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as fh:
            body = fh.read().decode("utf-8", "replace")
    except Exception:
        time.sleep(4.0)
        return
    if "slots available now" in body:
        return
    waits = [int(m) for m in re.findall(r"in (\d+) seconds", body)]
    if waits:
        time.sleep(min(max(min(waits) + 1, 1), max_wait))
    else:
        time.sleep(5.0)

# Central Madrid: the dense, mostly-flat-roofed core the model is calibrated
# for. Roughly Moncloa/Chamberi/Salamanca/Centro/Retiro/Arganzuela.
AOI_CORE = (40.3830, -3.7360, 40.4620, -3.6580)
# The whole municipality bounding box (includes a lot of empty land).
AOI_ALL = (40.3120, -3.8890, 40.5640, -3.5180)


def overpass(query: str, timeout: int = 300) -> dict:
    """POST a query, trying each mirror, with backoff on 429/504."""
    last = None
    for attempt in range(8):
        ep = ENDPOINTS[attempt % len(ENDPOINTS)]
        wait_for_slot()
        try:
            req = urllib.request.Request(
                ep, data=urllib.parse.urlencode({"data": query}).encode(),
                headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return json.load(fh)
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, json.JSONDecodeError, OSError) as exc:
            last = exc
            wait = min(120, 5 * 2 ** attempt)
            print(f"      retry in {wait}s ({type(exc).__name__})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"overpass failed after retries: {last}")


def tile_query(s: float, w: float, n: float, e: float) -> str:
    # `out geom` gives node coordinates inline, so no second .../way lookup.
    return (f"[out:json][timeout:240];"
            f'(way["building"]({s},{w},{n},{e});'
            f' way["building:part"]({s},{w},{n},{e}););'
            f"out geom tags;")


def fetch_tile(s, w, n, e, depth=0) -> list:
    """Fetch one bbox, splitting it if the server keeps refusing."""
    key = f"{s:.4f}_{w:.4f}_{n:.4f}_{e:.4f}.json"
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)["elements"]
    try:
        data = overpass(tile_query(s, w, n, e))
    except RuntimeError:
        if depth >= 2:
            print(f"      giving up on tile {key}", flush=True)
            return []
        # Too big or too busy: quarter it and try the pieces.
        mid_lat, mid_lon = (s + n) / 2, (w + e) / 2
        out = []
        for ss, ww, nn, ee in ((s, w, mid_lat, mid_lon),
                               (s, mid_lon, mid_lat, e),
                               (mid_lat, w, n, mid_lon),
                               (mid_lat, mid_lon, n, e)):
            out.extend(fetch_tile(ss, ww, nn, ee, depth + 1))
        return out
    els = [el for el in data.get("elements", []) if el.get("geometry")]
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"elements": els}, fh)
    return els


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-madrid", action="store_true",
                    help="the full municipality rather than the dense core")
    ap.add_argument("--step", type=float, default=0.010,
                    help="tile size in degrees")
    args = ap.parse_args()

    s0, w0, n0, e0 = AOI_ALL if args.all_madrid else AOI_CORE
    os.makedirs(CACHE, exist_ok=True)

    lat_steps = math.ceil((n0 - s0) / args.step)
    lon_steps = math.ceil((e0 - w0) / args.step)
    total = lat_steps * lon_steps
    print(f"AOI {s0},{w0} .. {n0},{e0}   {lat_steps}x{lon_steps} = "
          f"{total} tiles", flush=True)

    boxes = []
    for i in range(lat_steps):
        for j in range(lon_steps):
            s = s0 + i * args.step
            w = w0 + j * args.step
            boxes.append((s, w, min(s + args.step, n0), min(w + args.step, e0)))

    seen, elements, done = set(), [], 0
    t0 = time.time()
    pending = list(boxes)
    # Several sweeps: a tile the server refused outright is retried later
    # rather than being allowed to end the run. Earlier versions of this
    # script died three times at tiles 14, 18 and 22 for exactly that reason.
    for sweep in range(4):
        if not pending:
            break
        if sweep:
            print(f"\n  sweep {sweep + 1}: retrying "
                  f"{len(pending)} unfetched tiles", flush=True)
        failed = []
        for (s, w, n, e) in pending:
            cached = os.path.exists(os.path.join(
                CACHE, f"{s:.4f}_{w:.4f}_{n:.4f}_{e:.4f}.json"))
            try:
                els = fetch_tile(s, w, n, e)
            except Exception as exc:
                failed.append((s, w, n, e))
                print(f"  tile deferred ({type(exc).__name__})", flush=True)
                time.sleep(5.0)
                continue
            for el in els:
                if el["id"] not in seen:
                    seen.add(el["id"])
                    elements.append(el)
            done += 1
            rate = done / max(time.time() - t0, 1e-9)
            eta = (total - done) / rate if rate else 0
            print(f"  tile {done}/{total}  +{len(els):5d}  "
                  f"total {len(elements):7,d}  eta {eta/60:5.1f} min"
                  f"{'  (cached)' if cached else ''}", flush=True)
            if not cached:
                time.sleep(1.0)          # be kind to a donated endpoint
        pending = failed

    if pending:
        print(f"\n  {len(pending)} tiles could not be fetched and are "
              f"missing from the extract.", flush=True)

    out = os.path.join(HERE, "data", "city", "osm_city.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"elements": elements}, fh)
    mb = os.path.getsize(out) / 1e6
    print(f"\n{len(elements):,} unique buildings -> {out}  ({mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
