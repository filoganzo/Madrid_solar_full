"""
spatial.py — a uniform grid index over densified building outlines.

Why this exists. `shading.build_horizon_fast` filters the whole densified
point cloud for every panel, which is fine for one block (5,538 panels x
~48,000 points) and quadratic everywhere after that. At city scale the same
code would compare ~10^6 panels against ~4x10^6 points: about 10^13 distance
tests, which is not a slow program, it is a program that never finishes.

The fix is not an approximation. `build_horizon_fast` already discards every
point beyond `max_radius_m`, so indexing the cloud into square cells and
visiting only the cells within that radius returns *the same points* and
therefore the same horizon, bit for bit. `tests/test_city.py` asserts that
equality against the original implementation rather than trusting it.

The grid is deliberately the dumbest structure that works: no k-d tree, no
dependency, and cell lookup is integer arithmetic. Buildings are spread
roughly evenly across a city, which is exactly the case a uniform grid is
good at and the case where a tree's overhead buys nothing.
"""

from __future__ import annotations

import numpy as np


class PointGrid:
    """Bucket (N, 3) x/y/height points into square cells for radius queries."""

    def __init__(self, pts_xyh: np.ndarray, cell_m: float = 100.0):
        self.cell = float(cell_m)
        self.pts = np.asarray(pts_xyh, dtype=float)
        if len(self.pts) == 0:
            self.x0 = self.y0 = 0.0
            self.nx = self.ny = 1
            self.starts = np.zeros(2, dtype=np.int64)
            self.order = np.zeros(0, dtype=np.int64)
            return

        self.x0 = float(self.pts[:, 0].min())
        self.y0 = float(self.pts[:, 1].min())
        ix = ((self.pts[:, 0] - self.x0) / self.cell).astype(np.int64)
        iy = ((self.pts[:, 1] - self.y0) / self.cell).astype(np.int64)
        self.nx = int(ix.max()) + 1
        self.ny = int(iy.max()) + 1

        flat = iy * self.nx + ix
        # Counting sort into contiguous per-cell runs: `order` lists point
        # indices grouped by cell, `starts` says where each cell's run begins.
        self.order = np.argsort(flat, kind="stable")
        counts = np.bincount(flat, minlength=self.nx * self.ny)
        self.starts = np.zeros(self.nx * self.ny + 1, dtype=np.int64)
        np.cumsum(counts, out=self.starts[1:])

    def query(self, x: float, y: float, radius_m: float) -> np.ndarray:
        """Every point within `radius_m` of (x, y), as an (M, 3) array.

        Returns a superset-free result: cells are selected by their index
        range, then the caller's own distance test (which it does anyway)
        trims the corners of the cell block.
        """
        if len(self.pts) == 0:
            return self.pts

        lo_x = int((x - radius_m - self.x0) // self.cell)
        hi_x = int((x + radius_m - self.x0) // self.cell)
        lo_y = int((y - radius_m - self.y0) // self.cell)
        hi_y = int((y + radius_m - self.y0) // self.cell)
        lo_x = max(lo_x, 0); lo_y = max(lo_y, 0)
        hi_x = min(hi_x, self.nx - 1); hi_y = min(hi_y, self.ny - 1)
        if lo_x > hi_x or lo_y > hi_y:
            return self.pts[:0]

        runs = []
        for cy in range(lo_y, hi_y + 1):
            base = cy * self.nx
            # One contiguous slice per row of cells, not one per cell.
            s = self.starts[base + lo_x]
            e = self.starts[base + hi_x + 1]
            if e > s:
                runs.append(self.order[s:e])
        if not runs:
            return self.pts[:0]
        idx = runs[0] if len(runs) == 1 else np.concatenate(runs)
        return self.pts[idx]
