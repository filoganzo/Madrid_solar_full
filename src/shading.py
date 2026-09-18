"""
shading.py — the part that makes this a Madrid model instead of a spreadsheet.

Every rooftop-potential study I read does shading as a flat percentage loss.
That is fine for a field in Extremadura and wrong for the Ensanche, where the
loss is not a percentage but a shape: it is concentrated at the hours and the
seasons that decide the economics. A panel on the north edge of a courtyard
loses almost nothing in July and almost everything in December, and December
is when Spanish evening demand peaks.

So shading is computed geometrically, per panel, from real building
footprints and heights:

  build_horizon()      for a point on a roof, the elevation angle of the
                       skyline in every direction (1 deg bins)
  beam_mask()          hour by hour, is the sun above that skyline
  sky_view_factor()    how much of the diffuse sky the panel can still see,
                       weighted correctly for its own tilt

The three are kept separate because they fail differently. Beam shading is a
hard on/off that moves with the season. Sky-view loss is a constant haircut
that never goes away. Row self-shading is a design choice we control. Lumping
them into one number hides which of the three we can actually do something
about.

Coordinates: local ENU metres, x = east, y = north, z = up.
Azimuth convention matches solar_geometry: 0 = south, +west, -east.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_BINS = 360          # 1 degree azimuth resolution
EDGE_SAMPLE_M = 2.0   # densify building edges to this spacing


@dataclass
class Obstacle:
    """A building, as a horizontal footprint extruded to a height."""
    polygon: np.ndarray     # (N, 2) in metres, ENU
    height_m: float
    osm_id: int = -1
    kind: str = "building"


def _densify(poly: np.ndarray, spacing: float = EDGE_SAMPLE_M) -> np.ndarray:
    """Put a point every `spacing` metres along the footprint outline.

    Vertices alone are not enough: a 40 m facade sampled only at its corners
    lets the sun through the middle of a wall. This bug inflates winter yield
    by about 15% on courtyard roofs, which is the kind of error that survives
    an annual-total sanity check and still ruins the seasonal shape.
    """
    pts = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        seg = b - a
        d = float(np.hypot(*seg))
        k = max(int(np.ceil(d / spacing)), 1)
        for j in range(k):
            pts.append(a + seg * (j / k))
    return np.asarray(pts) if pts else poly


def build_horizon(point: np.ndarray, obstacles: list[Obstacle],
                  n_bins: int = N_BINS,
                  max_radius_m: float = 400.0) -> np.ndarray:
    """Elevation angle of the skyline, in degrees, per azimuth bin.

    point is (x, y, z) with z the height of the panel above local datum.
    Returns an (n_bins,) array; bin i covers azimuth
    -180 + i * 360/n_bins degrees.
    """
    horizon = np.zeros(n_bins)
    px, py, pz = point

    for ob in obstacles:
        if ob.height_m <= pz + 0.05:
            continue                      # cannot block anything above us
        pts = _densify(ob.polygon)
        dx = pts[:, 0] - px
        dy = pts[:, 1] - py
        dist = np.hypot(dx, dy)
        keep = (dist > 0.5) & (dist < max_radius_m)
        if not keep.any():
            continue
        dx, dy, dist = dx[keep], dy[keep], dist[keep]

        # Azimuth in the south-zero, west-positive convention.
        az = np.degrees(np.arctan2(dx, -dy))
        az = np.where(az > 180.0, az - 360.0, az)
        az = np.where(az < -180.0, az + 360.0, az)

        elev = np.degrees(np.arctan2(ob.height_m - pz, dist))

        idx = np.clip(((az + 180.0) / 360.0 * n_bins).astype(int), 0,
                      n_bins - 1)
        np.maximum.at(horizon, idx, elev)

    # A sampled outline can leave single empty bins between samples on a
    # distant facade. Fill them from their neighbours so the skyline is
    # continuous rather than combed.
    for _ in range(2):
        left = np.roll(horizon, 1)
        right = np.roll(horizon, -1)
        gap = (horizon < left - 3.0) & (horizon < right - 3.0)
        horizon = np.where(gap, np.minimum(left, right), horizon)

    return horizon


def add_parapet(horizon: np.ndarray, panel_xy: np.ndarray,
                roof_polygon: np.ndarray, parapet_h: float,
                panel_z_above_roof: float,
                n_bins: int = N_BINS) -> np.ndarray:
    """Add the roof's own parapet wall to the horizon.

    The antepecho is about a metre high and runs round every flat roof in
    Madrid. For a panel sitting 0.3 m off the deck two metres from the edge,
    it puts a 20 deg obstruction across a third of the sky. It is the most
    commonly ignored obstacle in rooftop studies and it is the reason the
    first row of panels is worth less than the second.
    """
    pts = _densify(roof_polygon, 1.0)
    dx = pts[:, 0] - panel_xy[0]
    dy = pts[:, 1] - panel_xy[1]
    dist = np.hypot(dx, dy)
    keep = dist > 0.3
    dx, dy, dist = dx[keep], dy[keep], dist[keep]

    az = np.degrees(np.arctan2(dx, -dy))
    az = np.where(az > 180.0, az - 360.0, az)
    az = np.where(az < -180.0, az + 360.0, az)

    rise = parapet_h - panel_z_above_roof
    if rise <= 0:
        return horizon
    elev = np.degrees(np.arctan2(rise, dist))
    idx = np.clip(((az + 180.0) / 360.0 * n_bins).astype(int), 0, n_bins - 1)
    out = horizon.copy()
    np.maximum.at(out, idx, elev)
    return out


def add_row_self_shading(horizon: np.ndarray, tilt_deg: float,
                         surface_az_deg: float, panel_length_m: float,
                         row_pitch_m: float, n_bins: int = N_BINS,
                         spread_deg: float = 70.0) -> np.ndarray:
    """Add the row in front of this one.

    On a flat roof, tilting panels up costs area. Steeper tilt earns more per
    panel and fits fewer panels. The trade is settled by row pitch, not by
    tilt, which is why config sets 15 deg rather than the 37 deg optimum: at
    37 deg the pitch needed to keep the winter rows clear is 2.5x the panel
    length, and the roof runs out of space before the panels run out of sun.
    run_madrid.py tests both and prints the comparison.
    """
    h = panel_length_m * np.sin(np.radians(tilt_deg))
    if row_pitch_m <= 0 or h <= 0:
        return horizon
    elev = np.degrees(np.arctan2(h, row_pitch_m))

    az_bins = np.linspace(-180.0, 180.0, n_bins, endpoint=False)
    delta = np.abs(((az_bins - surface_az_deg + 180.0) % 360.0) - 180.0)
    mask = delta < spread_deg
    out = horizon.copy()
    out[mask] = np.maximum(out[mask], elev)
    return out


def beam_mask(horizon: np.ndarray, sun_alt_deg, sun_az_deg,
              n_bins: int = N_BINS):
    """1 where the sun clears the skyline, 0 where a building is in the way.

    Deliberately binary. A partial-shading model would be more accurate for a
    single string, but it would also be false precision: with a real inverter
    the electrical consequence of clipping one cell row is not proportional
    to the shaded area, and modelling that properly needs module-level data
    I do not have. Binary and stated is better than smooth and invented.
    """
    idx = np.clip(((np.asarray(sun_az_deg) + 180.0) / 360.0 * n_bins)
                  .astype(int), 0, n_bins - 1)
    return np.where(np.asarray(sun_alt_deg) > horizon[idx], 1.0, 0.0)


def sky_view_factor(horizon: np.ndarray, tilt_deg: float,
                    surface_az_deg: float, n_alt: int = 90) -> float:
    """Fraction of the isotropic diffuse sky the panel still sees.

    Numeric integration of cos(incidence) over the visible hemisphere,
    normalised by the same integral with no obstruction. Weighted by the
    panel's own tilt, because a tilted panel already ignores half the sky and
    the half it ignores is not the half the buildings take.
    """
    n_bins = len(horizon)
    az = np.radians(np.linspace(-180.0, 180.0, n_bins, endpoint=False))
    alt = np.radians(np.linspace(0.5, 89.5, n_alt))
    AZ, ALT = np.meshgrid(az, alt, indexing="ij")

    tilt = np.radians(tilt_deg)
    saz = np.radians(surface_az_deg)
    cos_i = (np.sin(ALT) * np.cos(tilt)
             + np.cos(ALT) * np.sin(tilt) * np.cos(AZ - saz))
    cos_i = np.clip(cos_i, 0.0, None)

    d_omega = np.cos(ALT)
    weight = cos_i * d_omega

    visible = ALT > np.radians(horizon)[:, None]
    total = weight.sum()
    if total <= 0:
        return 0.0
    return float((weight * visible).sum() / total)


def horizon_summary(horizon: np.ndarray) -> dict:
    """Numbers a person can sanity-check by looking out of a window."""
    az = np.linspace(-180.0, 180.0, len(horizon), endpoint=False)
    south = np.abs(az) < 45.0
    return {
        "mean_elev_deg": float(horizon.mean()),
        "max_elev_deg": float(horizon.max()),
        "south_sector_mean_deg": float(horizon[south].mean()),
        "share_sky_blocked": float((horizon > 5.0).mean()),
    }


# --------------------------------------------------------------------------
# Vectorised variant
# --------------------------------------------------------------------------

def densify_all(obstacles: list[Obstacle],
                spacing: float = EDGE_SAMPLE_M) -> np.ndarray:
    """Pre-densify every footprint once into an (N, 3) array of x, y, height.

    Called once per run instead of once per panel. With ~1,500 panel
    positions and ~100 buildings this is the difference between a 40-minute
    run and a 20-second one, which matters because a model you have to wait
    40 minutes for is a model you stop re-running, and a model you stop
    re-running stops being checked.
    """
    chunks = []
    for ob in obstacles:
        pts = _densify(ob.polygon, spacing)
        h = np.full((len(pts), 1), ob.height_m)
        chunks.append(np.hstack([pts, h]))
    return np.vstack(chunks) if chunks else np.zeros((0, 3))


def build_horizon_fast(point: np.ndarray, pts_xyh: np.ndarray,
                       n_bins: int = N_BINS,
                       max_radius_m: float = 400.0) -> np.ndarray:
    """Same result as build_horizon, computed with numpy over pre-densified
    points. tests/test_sanity.py asserts the two agree."""
    px, py, pz = point
    dx = pts_xyh[:, 0] - px
    dy = pts_xyh[:, 1] - py
    dist = np.hypot(dx, dy)
    rise = pts_xyh[:, 2] - pz
    keep = (dist > 0.5) & (dist < max_radius_m) & (rise > 0.05)
    if not keep.any():
        return np.zeros(n_bins)
    dx, dy, dist, rise = dx[keep], dy[keep], dist[keep], rise[keep]

    az = np.degrees(np.arctan2(dx, -dy))
    az = np.where(az > 180.0, az - 360.0, az)
    az = np.where(az < -180.0, az + 360.0, az)
    elev = np.degrees(np.arctan2(rise, dist))

    horizon = np.zeros(n_bins)
    idx = np.clip(((az + 180.0) / 360.0 * n_bins).astype(int), 0, n_bins - 1)
    np.maximum.at(horizon, idx, elev)

    for _ in range(2):
        left = np.roll(horizon, 1)
        right = np.roll(horizon, -1)
        gap = (horizon < left - 3.0) & (horizon < right - 3.0)
        horizon = np.where(gap, np.minimum(left, right), horizon)
    return horizon
