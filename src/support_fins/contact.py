"""Contact lines: the polyline a wall's top follows under an overhang, and the
passes that bring it to exactly `gap` below the part.

Ported from web/prop/contact.js and web/prop/surface.js in
https://github.com/gittrahan/support-fins (MIT). The JS comments carry the
measurements behind each pass; the short versions are kept here.
"""
from __future__ import annotations

import numpy as np
import trimesh


class Surface:
    """'What part surface is directly above (x, y)?' for a set of triangles
    (a region, or the whole part), via vertical rays."""

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.z0 = mesh.bounds[0][2] - 1.0

    def z_at(self, xy) -> np.ndarray:
        """Lowest surface z above each (x, y); NaN where the column misses."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        out = np.full(len(xy), np.inf)
        if len(xy):
            origins = np.column_stack([xy, np.full(len(xy), self.z0)])
            dirs = np.tile([0.0, 0.0, 1.0], (len(xy), 1))
            locs, ri, _ = self.mesh.ray.intersects_location(origins, dirs, multiple_hits=True)
            if len(ri):
                np.minimum.at(out, ri, locs[:, 2])
        out[np.isinf(out)] = np.nan
        return out

    def zs_at(self, xy):
        """Every surface z above each (x, y), as a list of arrays.

        `z_at` keeps only the lowest hit, which is what a bed-attached wall wants.
        A part-attached support instead needs the surfaces IN BETWEEN -- the floor
        it stands on lives above the plate and below the overhang -- so keep them
        all (see `floor_line`)."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        out = [[] for _ in range(len(xy))]
        if len(xy):
            origins = np.column_stack([xy, np.full(len(xy), self.z0)])
            dirs = np.tile([0.0, 0.0, 1.0], (len(xy), 1))
            locs, ri, _ = self.mesh.ray.intersects_location(origins, dirs, multiple_hits=True)
            locs = np.asarray(locs)
            if locs.ndim == 1:
                locs = locs.reshape(-1, 3)
            for z, i in zip(locs[:, 2], np.asarray(ri)):
                out[i].append(z)
        return [np.array(a) for a in out]



def across(line) -> np.ndarray:
    """Unit horizontal vector across the wall at each station (local tangent x Z);
    zero where the tangent is degenerate."""
    line = np.asarray(line)
    i = np.arange(len(line))
    a = line[np.maximum(i - 1, 0), :2]
    b = line[np.minimum(i + 1, len(line) - 1), :2]
    r = b - a
    n = np.linalg.norm(r, axis=1, keepdims=True)
    r = np.divide(r, n, out=np.zeros_like(r), where=n > 1e-9)
    return np.column_stack([r[:, 1], -r[:, 0]])


def straightness(line) -> float:
    """RMS perpendicular deviation from the best-fit XY axis, over the chord.

    A precondition, not a quality score: a bowl's lowest points form a ring, and
    a wall swept along it saws through the part. Chord-normalised so the score
    doesn't drift with station density (arc/chord did)."""
    xy = np.asarray(line)[:, :2]
    if len(xy) < 3:
        return np.inf
    c = xy - xy.mean(axis=0)
    ax = np.linalg.svd(c, full_matrices=False)[2][0]
    t = c @ ax
    perp = c @ np.array([-ax[1], ax[0]])
    chord = t.max() - t.min()
    return np.inf if chord < 1e-9 else float(np.sqrt((perp ** 2).mean()) / chord)


def contact_line(pts, surface: Surface, step: float, tip: float):
    """Lowest-underside polyline of a region: principal horizontal axis, buckets
    `step` mm apart along it, min-z sample per bucket -- then re-fit each station
    to the surface directly above its OWN xy (the bucket's lowest point is
    usually somewhere off to the side, which is what buried the spike's walls)."""
    pts = np.asarray(pts)
    xy = pts[:, :2] - pts[:, :2].mean(axis=0)
    if np.allclose(xy, 0):
        return None
    axis = np.linalg.svd(xy, full_matrices=False)[2][0]
    t = xy @ axis
    lo, hi = t.min(), t.max()
    if hi - lo < 1e-6:
        return None
    n = max(3, int(np.ceil((hi - lo) / step)))
    k = np.minimum(((t - lo) / (hi - lo) * n).astype(int), n - 1)
    line = []
    for b in range(n):
        sel = pts[k == b]
        if len(sel):
            line.append(sel[np.argmin(sel[:, 2])])
    if len(line) < 3:
        return None
    line = np.array(line, dtype=float)

    zz = surface.z_at(line[:, :2])
    ok = ~np.isnan(zz)
    line[ok, 2] = zz[ok]

    contour_top(line, surface, tip)
    lower_sag(line, surface)
    return line


def lower_sag(line, surface: Surface, passes: int = 3, band: float = np.inf):
    """Pull down both ends of any span whose straight top would pass above the
    surface at its midpoint (measured 0.005 mm of clearance where 0.2 was meant).

    `band` caps how far below the wall a surface may be and still pull it down: a
    part-attached top must settle against the OVERHANG, not the floor beneath it."""
    mid = (line[:-1, :2] + line[1:, :2]) / 2
    zz = surface.z_at(mid)                       # xy never moves, so query once
    for _ in range(passes):
        moved = False
        for i in range(len(line) - 1):
            if np.isnan(zz[i]):
                continue
            over = (line[i, 2] + line[i + 1, 2]) / 2 - zz[i]
            if over > band:
                continue
            if over > 1e-4:
                line[i, 2] -= over
                line[i + 1, 2] -= over
                moved = True
        if not moved:
            break
    return line


def contour_top(line, surface: Surface, tip: float, band: float = np.inf):
    """Lower each station until both corners of its tip-wide top clear the part,
    not just the centre. Past ~34deg of slope the up-slope corner otherwise sits
    inside the part. Across the tip only, never along the run.

    `band` restricts the pull-down to surfaces within that distance of the
    overhang: past it, the lowest hit is the floor a part-attached wall lands on,
    not the overhang it clears."""
    s = across(line)
    half = tip / 2
    for o in (-half, half):
        zz = surface.z_at(line[:, :2] + s * o)
        low = (~np.isnan(zz) & (zz < line[:, 2]) & (line[:, 2] - zz <= band)
               & np.any(s != 0, axis=1))
        line[low, 2] = zz[low]
    return line


def settle_top(line, surface: Surface, gap: float, tip: float,
               step: float = 0.25, passes: int = 3, band: float = np.inf):
    """Measure the built top edge every `step` mm (both tip corners + centre) and
    lower each station by what its own span lacks, so the closest approach is
    exactly `gap`. Lower only, never raise: a larger clearance usually means the
    nearest thing is outside this region, which this pass can't see. `band` skips
    surfaces more than that far below the wall top (the floor under a
    part-attached wall, not its overhang)."""
    half = tip / 2
    xs, owner, seg, tpar = [], [], [], []
    for i in range(len(line) - 1):
        p, q = line[i], line[i + 1]
        d = q[:2] - p[:2]
        L = np.hypot(*d)
        if L < 1e-9:
            continue
        sx = np.array([d[1], -d[0]]) / L
        n = max(1, int(np.ceil(L / step)))
        for k in range(n + 1):
            t = k / n
            for o in (-half, 0.0, half):
                xs.append(p[:2] + d * t + sx * o)
                seg.append(i)
                tpar.append(t)
                owner.append(i if t < 0.5 else i + 1)
    if not xs:
        return line
    zz = surface.z_at(np.array(xs))
    seg, tpar, owner = np.array(seg), np.array(tpar), np.array(owner)
    ok = ~np.isnan(zz)
    seg, tpar, owner, zz = seg[ok], tpar[ok], owner[ok], zz[ok]
    for _ in range(passes):
        top = line[seg, 2] + (line[seg + 1, 2] - line[seg, 2]) * tpar - gap
        far = (top - zz) > band
        need = np.where(far, 0.0, gap - (zz - top))
        drop = np.zeros(len(line))
        np.maximum.at(drop, owner, np.where(need > 1e-4, need, 0.0))
        if not drop.any():
            break
        line[:, 2] -= drop
    return line

