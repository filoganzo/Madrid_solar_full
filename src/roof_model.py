"""
roof_model.py — turning a city block into a list of panel positions.

Three jobs:
  1. Project WGS84 lat/lon to local metres, so all geometry is Euclidean.
  2. Give every building a height, and record where that height came from.
  3. Pack panels onto each flat roof, respecting setbacks, rooftop plant and
     row pitch.

On job 2, the honest finding first: of the 102 buildings in the extract, 6
carry building:levels in OpenStreetMap. Six. So 94% of the heights in this
model are inferred from typology, and heights are what drive mutual shading.
That makes building height the binding data gap, not a detail — which is why
every building carries a `height_source` field that the map renders, so a
reader can see exactly which roofs are measured and which are guessed.

The fix is not more OSM. It is Catastro INSPIRE (which publishes floor counts
per building for all of Spain) joined to PNOA LiDAR (0.5 m vertical, free
from IGN). That is a two-day job and it is the first thing I would do with a
real budget. Until then the model states its own ignorance rather than
hiding it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from .shading import Obstacle

R_EARTH = 6378137.0


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------

class LocalENU:
    """Equirectangular projection about a local origin.

    Over a 500 m block the distortion against a proper UTM projection is
    under 2 cm. Using a full projection library here would add a dependency
    for accuracy the model cannot use.
    """

    def __init__(self, lat0: float, lon0: float):
        self.lat0 = lat0
        self.lon0 = lon0
        self.mx = math.pi / 180.0 * R_EARTH * math.cos(math.radians(lat0))
        self.my = math.pi / 180.0 * R_EARTH

    def to_xy(self, lat, lon):
        x = (np.asarray(lon) - self.lon0) * self.mx
        y = (np.asarray(lat) - self.lat0) * self.my
        return x, y

    def to_latlon(self, x, y):
        lon = np.asarray(x) / self.mx + self.lon0
        lat = np.asarray(y) / self.my + self.lat0
        return lat, lon


# --------------------------------------------------------------------------
# Buildings
# --------------------------------------------------------------------------

RESIDENTIAL_TAGS = {"yes", "residential", "apartments", "house", "terrace",
                    "dormitory", "detached"}
NON_ROOF_TAGS = {"roof", "carport", "garage", "garages", "shed", "hut",
                 "construction", "ruins"}


@dataclass
class Building:
    osm_id: int
    polygon: np.ndarray            # (N,2) metres
    height_m: float
    height_source: str             # "osm:height" | "osm:levels" | "inferred"
    levels: int
    use: str                       # "residential" | "office" | "other"
    tags: dict = field(default_factory=dict)
    area_m2: float = 0.0
    flat_roof: bool = True

    @property
    def is_addressable(self) -> bool:
        """Residential, flat-roofed, and big enough that a job on it is worth
        mobilising for. The area floor is a commercial judgement, not a
        physical one: below roughly 120 m2 of footprint the fixed cost of
        roof access dominates whatever the panels earn."""
        return (self.use == "residential" and self.flat_roof
                and self.area_m2 >= 120.0)


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def infer_height(tags: dict, area_m2: float, cfg: dict) -> tuple[float, str, int]:
    """Height, its provenance, and floor count.

    Priority: measured height > tagged floor count > typology prior.

    The typology prior for the Madrid Ensanche: the 1860-1930 grid was built
    to a cornice line, so heights cluster tightly around 5-7 storeys. Large
    footprints on the Castellana axis are post-1960 offices and are much
    taller. Encoding that as a rule rather than a single average matters
    because a 15-storey tower on the west side of a block removes the
    afternoon from every roof behind it.
    """
    storey = cfg["roof"]["storey_height_m"]
    parapet = cfg["roof"]["parapet_height_m"]

    if tags.get("height"):
        try:
            h = float(str(tags["height"]).split()[0])
            return h, "osm:height", max(int(round(h / storey)), 1)
        except ValueError:
            pass

    if tags.get("building:levels"):
        try:
            lv = int(float(tags["building:levels"]))
            return lv * storey + parapet, "osm:levels", lv
        except ValueError:
            pass

    use = tags.get("building", "yes")
    if use in ("office", "commercial", "retail") and area_m2 > 900:
        lv = 12                       # Castellana office typology
    elif use in ("office", "commercial", "retail"):
        lv = 7
    elif area_m2 < 150:
        lv = 2                        # mews, garages, courtyard annexes
    else:
        lv = cfg["roof"]["levels_default"]
    return lv * storey + parapet, "inferred", lv


def classify_use(tags: dict) -> str:
    b = tags.get("building", "yes")
    if b in RESIDENTIAL_TAGS:
        return "residential"
    if b in ("office", "commercial", "retail"):
        return "office"
    return "other"


def load_osm(path: str, cfg: dict) -> tuple[list[Building], LocalENU]:
    """Read the Overpass extract and build the geometry list."""
    raw = json.load(open(path))
    ways = [e for e in raw["elements"]
            if e.get("type") == "way" and e.get("geometry")]

    lats = [p["lat"] for w in ways for p in w["geometry"]]
    lons = [p["lon"] for w in ways for p in w["geometry"]]
    proj = LocalENU(float(np.mean(lats)), float(np.mean(lons)))

    buildings: list[Building] = []
    for w in ways:
        tags = w.get("tags", {}) or {}
        if tags.get("building") in NON_ROOF_TAGS:
            continue
        lat = np.array([p["lat"] for p in w["geometry"]])
        lon = np.array([p["lon"] for p in w["geometry"]])
        x, y = proj.to_xy(lat, lon)
        poly = np.column_stack([x, y])
        # Overpass closes ways by repeating the first node.
        if len(poly) > 2 and np.allclose(poly[0], poly[-1]):
            poly = poly[:-1]
        if len(poly) < 3:
            continue
        area = polygon_area(poly)
        if area < 25.0:
            continue
        h, src, lv = infer_height(tags, area, cfg)
        roof_shape = tags.get("roof:shape", "")
        flat = roof_shape in ("", "flat") or roof_shape == "flat"
        buildings.append(Building(
            osm_id=w["id"], polygon=poly, height_m=h, height_source=src,
            levels=lv, use=classify_use(tags), tags=tags, area_m2=area,
            flat_roof=flat,
        ))
    return buildings, proj


def to_obstacles(buildings: list[Building]) -> list[Obstacle]:
    return [Obstacle(polygon=b.polygon, height_m=b.height_m, osm_id=b.osm_id,
                     kind=b.use) for b in buildings]


# --------------------------------------------------------------------------
# Panel packing
# --------------------------------------------------------------------------

def point_in_polygon(px: float, py: float, poly: np.ndarray) -> bool:
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def dist_to_boundary(px: float, py: float, poly: np.ndarray) -> float:
    """Shortest distance from a point to the footprint outline.

    Used instead of a polygon-offset algorithm to enforce the setback: a
    panel is allowed only if all four of its corners are at least
    `setback_m` from the roof edge. Same effect, no dependency, and it
    handles the concave courtyard shapes of the Ensanche correctly, which a
    naive inward offset does not.
    """
    best = float("inf")
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ab = b - a
        t = 0.0
        denom = float(ab @ ab)
        if denom > 1e-12:
            t = float(np.clip(((np.array([px, py]) - a) @ ab) / denom, 0, 1))
        proj = a + t * ab
        d = math.hypot(px - proj[0], py - proj[1])
        best = min(best, d)
    return best


@dataclass
class PanelSlot:
    x: float
    y: float
    z: float                 # height above datum of the panel mid-point
    building_id: int
    row: int
    col: int


def plant_block(poly: np.ndarray, share: float) -> np.ndarray | None:
    """A rectangle at the roof centroid standing in for stairhead, lift
    overrun, water tanks, HVAC and washing lines.

    Represented as one solid keep-out rather than scattered across the roof
    because that is how it actually sits on a Madrid azotea: the castillete
    is one structure over the stair core. Its position matters as much as its
    area, since a central block shades the rows north of it.
    """
    if share <= 0:
        return None
    a = polygon_area(poly) * share
    cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
    side = math.sqrt(a)
    half = side / 2.0
    return np.array([[cx - half, cy - half], [cx + half, cy - half],
                     [cx + half, cy + half], [cx - half, cy + half]])


def pack_panels(b: Building, cfg: dict) -> tuple[list[PanelSlot], np.ndarray | None]:
    """Lay panels out in rows on a flat roof.

    Rows run east-west (perpendicular to the array azimuth) so that each row
    shades only the one behind it. Spacing across rows is row_pitch_factor x
    the panel's projected height, sized so the rows stay clear at the winter
    solstice between 10:00 and 14:00 — the window that actually matters,
    because chasing zero shading at 08:00 in December costs 30% of the roof
    for 2% of the annual yield.
    """
    p = cfg["panel"]
    r = cfg["roof"]
    setback = r["setback_m"]
    tilt = math.radians(r["tilt_deg"])

    # Panel in landscape: `length` runs across the row, `width` up the slope.
    across = p["length_m"] + 0.02            # clamp gap
    up_slope = p["width_m"] * math.cos(tilt)  # horizontal footprint
    pitch = up_slope * r["row_pitch_factor"]

    az = math.radians(r["azimuth_deg"])
    # Unit vectors: `u` along a row, `v` from one row to the next (northward
    # component of the array azimuth).
    u = np.array([math.cos(az), -math.sin(az)])
    v = np.array([math.sin(az), math.cos(az)])

    keepout = plant_block(b.polygon, r["obstruction_share"])

    minx, maxx = b.polygon[:, 0].min(), b.polygon[:, 0].max()
    miny, maxy = b.polygon[:, 1].min(), b.polygon[:, 1].max()
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    reach = math.hypot(maxx - minx, maxy - miny) / 2.0 + 2.0

    n_rows = int(reach * 2 / pitch) + 1
    n_cols = int(reach * 2 / across) + 1

    slots: list[PanelSlot] = []
    panel_z = b.height_m - r["parapet_height_m"] + 0.35  # deck + frame foot

    for i in range(-n_rows, n_rows + 1):
        for j in range(-n_cols, n_cols + 1):
            c = np.array([cx, cy]) + v * (i * pitch) + u * (j * across)
            corners = [c + u * (across / 2) * sx + v * (up_slope / 2) * sy
                       for sx in (-1, 1) for sy in (-1, 1)]
            ok = True
            for q in corners:
                if not point_in_polygon(q[0], q[1], b.polygon):
                    ok = False
                    break
                if dist_to_boundary(q[0], q[1], b.polygon) < setback:
                    ok = False
                    break
                if keepout is not None and point_in_polygon(q[0], q[1],
                                                            keepout):
                    ok = False
                    break
            if ok:
                slots.append(PanelSlot(float(c[0]), float(c[1]), panel_z,
                                       b.osm_id, i, j))

    # Access walkways: every Nth row is given up to maintenance routes.
    if slots and r["walkway_share"] > 0:
        rows = sorted({s.row for s in slots})
        keep_every = max(int(round(1.0 / r["walkway_share"])), 2)
        drop = {row for k, row in enumerate(rows) if k % keep_every == 0}
        if len(drop) < len(rows):
            slots = [s for s in slots if s.row not in drop]

    return slots, keepout
