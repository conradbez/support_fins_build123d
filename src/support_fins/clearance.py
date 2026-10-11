"""Which stations may carry a wall, and whether a finished wall welds to the part.

Ported from web/prop/clearance.js, web/inside.js and the solidClearance check in
web/prop.js (https://github.com/gittrahan/support-fins, MIT). Rule throughout:
"no wall is a fixable disappointment, a fused wall is a ruined print".
"""
from __future__ import annotations

import numpy as np
import trimesh

from .contact import across


class Part:
    """Containment and nearest-surface queries against the whole seated part."""

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.prox = trimesh.proximity.ProximityQuery(mesh)

    def inside(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        return self.mesh.contains(pts) if len(pts) else np.zeros(0, bool)

    def nearest(self, pts):
        """(distance, cosUp) per point. cosUp is how far the approach to the part
        points up. Points inside the part get distance 0."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        closest, d, tri = self.prox.on_surface(pts)
        v = closest - pts
        cos_up = np.divide(v[:, 2], d, out=np.zeros_like(d), where=d > 1e-9)
        # inside when the part's face normal points the same way we must travel
        # to reach it
        inside = np.einsum('ij,ij->i', v, self.mesh.face_normals[tri]) > 1e-9
        d = np.where(inside, 0.0, d)
        return d, cos_up


def welds(d, cos_up, gap: float):
    """Would a wall this close weld? From ABOVE it is the breakaway interface,
    where a vertical gap of `gap` measures only ~gap*cos(slope), so the floor is
    0.675*gap (~47.5deg of slope). Sideways it is a flank and needs > 0.205 mm."""
    return np.where(cos_up > 0.7, d < gap * 0.675, d < 0.205)


def stations_clear(line, part: Part, profile_half, p) -> np.ndarray:
    """Can each station's wall reach the plate through open air? Probes the full
    height, at the wall's half-width PLUS a margin stepped through the side
    clearance corridor (a rib thinner than the corridor can hide between a
    single far probe and the wall face), and the centreline."""
    s = across(line)
    pts, owner = [], []
    for k in range(len(line)):
        if not s[k].any():
            continue
        top = line[k, 2] - p.gap
        n = max(12, int(np.ceil(top / 1.5)))
        for i in range(1, n + 1):
            z = top * i / n - 0.05
            if z <= 0:
                continue
            half = profile_half(z, top)
            xy = line[k, :2]
            pts.append([xy[0], xy[1], z])
            owner.append(k)
            for m in (0.12, 0.24, p.side_clear):
                for sign in (1, -1):
                    q = xy + s[k] * sign * (half + m)
                    pts.append([q[0], q[1], z])
                    owner.append(k)
    ok = np.ones(len(line), bool)
    if pts:
        hit = part.inside(np.array(pts))
        ok[np.array(owner)[hit]] = False
    return ok


def stations_certified(line, part: Part, profile_half, p) -> np.ndarray:
    """After settling: measure each station's cross-section outline (top flat,
    then both flanks every 2 mm) against the WHOLE part, the way the checker
    will. Catches the geometry settle_top couldn't see: another region, a face
    too steep to be an overhang, a surface arriving tangent to a flank."""
    s = across(line)
    pts, owner = [], []
    for k in range(len(line)):
        if not s[k].any():
            continue
        top = line[k, 2] - p.gap
        probes = [(0.0, top), (p.tip / 2, top), (-p.tip / 2, top)]
        for z in np.arange(0.4, top - 0.05, 2.0):
            h = profile_half(z, top)
            probes += [(h, z), (-h, z)]
        for o, z in probes:
            if z <= 0.05:
                continue
            q = line[k, :2] + s[k] * o
            pts.append([q[0], q[1], z])
            owner.append(k)
    ok = np.ones(len(line), bool)
    if pts:
        d, cu = part.nearest(np.array(pts))
        bad = welds(d, cu, p.gap)
        ok[np.array(owner)[bad]] = False
    return ok


def solid_clearance(fin: trimesh.Trimesh, part: Part, gap: float, density: float = 6.0):
    """Measure the finished solid: sample its surface (vertices + ~`density`
    points per mm^2) and return the xy of the worst welding point, or None.
    Stations are probed at stations; the last welds sat between them."""
    n = int(min(60000, max(500, fin.area * density)))
    pts, _ = trimesh.sample.sample_surface(fin, n, seed=0)
    pts = np.vstack([fin.vertices, pts])
    d, cu = part.nearest(pts)
    bad = welds(d, cu, gap)
    if not bad.any():
        return None
    return pts[bad][np.argmin(d[bad]), :2]


def longest_run(usable):
    """Longest contiguous run of True, as (a, b) half-open, or None. Trimming to
    it instead of discarding the whole line over one bad station is what lets a
    wall survive a local problem."""
    best, start = None, -1
    for i in range(len(usable) + 1):
        if i < len(usable) and usable[i]:
            if start < 0:
                start = i
        elif start >= 0:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = -1
    return best
