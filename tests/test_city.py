"""
test_city.py — the checks that let the city run be trusted.

The city model makes exactly one claim beyond what run_madrid.py already
claims: that running the same physics over 10^5 buildings instead of 10^2
changes the speed and nothing else. That claim has a precise form — the
spatial index must return the same horizon as the brute-force scan it
replaces — and it is the kind of claim that is easy to assert and easy to
have quietly wrong, because an index that drops a few distant points still
produces plausible-looking output.

So the first two tests compare the indexed horizon against the original
implementation directly, on real geometry, and require equality rather than
closeness. If someone later tunes the cell size or the query radius for
speed, these fail rather than silently degrading every yield in Madrid.
"""

import os
import sys

import numpy as np
import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src import roof_model, shading                      # noqa: E402
from src.spatial import PointGrid                        # noqa: E402


@pytest.fixture(scope="module")
def block():
    with open(os.path.join(ROOT, "config", "madrid.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    buildings, proj = roof_model.load_osm(
        os.path.join(ROOT, "data", "osm_raw.json"), cfg)
    pts = shading.densify_all(roof_model.to_obstacles(buildings))
    return cfg, buildings, proj, pts


# --------------------------------------------------------------------------
# The index must not change the answer
# --------------------------------------------------------------------------

def test_grid_returns_every_point_within_radius(block):
    """The grid is a filter, not an approximation: for a sample of roof
    points it must return a superset of everything inside the radius."""
    _, _, _, pts = block
    grid = PointGrid(pts, 100.0)
    rng = np.random.default_rng(7)
    for _ in range(25):
        x = float(rng.uniform(pts[:, 0].min(), pts[:, 0].max()))
        y = float(rng.uniform(pts[:, 1].min(), pts[:, 1].max()))
        r = 400.0
        got = grid.query(x, y, r)
        want = pts[np.hypot(pts[:, 0] - x, pts[:, 1] - y) <= r]
        # Every genuinely-near point is present (the grid may return extras
        # from the corners of the cell block; the caller's distance test
        # removes those).
        assert len(got) >= len(want)
        got_set = {tuple(np.round(p, 6)) for p in got}
        for p in want:
            assert tuple(np.round(p, 6)) in got_set


def test_indexed_horizon_identical_to_full_scan(block):
    """The claim the whole city run rests on, checked on real panels."""
    cfg, buildings, _, pts = block
    grid = PointGrid(pts, 100.0)
    addressable = [b for b in buildings if b.is_addressable]
    assert addressable, "fixture produced no addressable roofs"

    worst = 0.0
    checked = 0
    for b in addressable[:15]:
        slots, _ = roof_model.pack_panels(b, cfg)
        for s in slots[:8]:
            p = np.array([s.x, s.y, s.z])
            full = shading.build_horizon_fast(p, pts)
            span = float(np.hypot(np.ptp(b.polygon[:, 0]),
                                  np.ptp(b.polygon[:, 1])))
            local = grid.query(float(np.mean(b.polygon[:, 0])),
                               float(np.mean(b.polygon[:, 1])), 400.0 + span)
            idx = shading.build_horizon_fast(p, local)
            worst = max(worst, float(np.abs(full - idx).max()))
            checked += 1
    assert checked > 50, "not enough panels exercised"
    assert worst == 0.0, f"indexed horizon diverged by {worst} deg"


def test_grid_handles_empty_and_single_point():
    """Degenerate inputs must not raise: a tile at the edge of the AOI can
    legitimately contain nothing."""
    empty = PointGrid(np.zeros((0, 3)), 100.0)
    assert len(empty.query(0.0, 0.0, 400.0)) == 0
    one = PointGrid(np.array([[10.0, 20.0, 5.0]]), 100.0)
    assert len(one.query(10.0, 20.0, 50.0)) == 1
    assert len(one.query(9_000.0, 9_000.0, 50.0)) == 0


# --------------------------------------------------------------------------
# The city runner's own arithmetic
# --------------------------------------------------------------------------

def test_dwelling_estimate_matches_block_model():
    """run_city.estimate_dwellings must stay identical to the rule in
    run_madrid.py; it is the denominator of the headline result."""
    import run_city
    import run_madrid

    class FakeBuilding:
        def __init__(self, area, levels):
            self.area_m2 = area
            self.levels = levels

    with open(os.path.join(ROOT, "config", "madrid.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for area, levels in [(300, 6), (1200, 8), (150, 2), (900, 1)]:
        assert (run_city.estimate_dwellings(area, levels)
                == run_madrid.estimate_dwellings(FakeBuilding(area, levels),
                                                 cfg))


def test_city_sky_matches_block_sky():
    """The city run must use the same sky as the study, or its yields are not
    comparable with anything in the write-up."""
    import run_city
    import run_madrid

    with open(os.path.join(ROOT, "config", "madrid.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    doy, hour = run_madrid.build_time_grid()
    sun_b, _, _, sky_b, _ = run_madrid.calibrate(cfg, doy, hour)
    sun_c, _, _, sky_c = run_city.build_sky(cfg)

    assert np.allclose(sun_b.altitude_deg, sun_c.altitude_deg, atol=1e-9)
    assert np.allclose(sun_b.azimuth_deg, sun_c.azimuth_deg, atol=1e-9)
    for a, b in zip(sky_b, sky_c):
        assert np.allclose(a, b, atol=1e-9)
