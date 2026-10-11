"""Where the placer puts walls under a region: `split_region` cuts it into
locally-straight sub-patches, `patch_tracks` lays straight parallel tracks across
each.

Ported from web/prop/tracks.js in https://github.com/gittrahan/support-fins (MIT).
The JS comments carry the measurements; the short versions are kept here.
"""
from __future__ import annotations

import math

import numpy as np

from .clearance import Part
from .contact import Surface, contact_line, straightness


def split_region(mesh, faces, agree_deg: float = 15.0):
    """Split one overhang region into locally-straight sub-patches.

    A connected overhang region is a topological artifact, not a support unit: a
    tilted underside can be one region whose contact line sits far off any straight
    axis, so the straightness gate refuses its genuinely straight ledges with it.
    The unit a wall wants is the locally-flat piece.

    Region-grow from a seed face, judging every candidate against the SEED's
    normal (never its neighbour's: pairwise agreement creeps, so a cylinder merges
    into one "flat" patch). Seeds are largest-face-first."""
    faces = np.asarray(faces)
    normals = mesh.face_normals
    area = mesh.area_faces
    adj = mesh.face_adjacency
    fset = set(int(f) for f in faces)
    nbrs = {int(f): [] for f in faces}
    for a, b in adj:
        a, b = int(a), int(b)
        if a in fset and b in fset:
            nbrs[a].append(b)
            nbrs[b].append(a)

    cut = math.cos(math.radians(agree_deg))
    order = sorted(fset, key=lambda f: -area[f])
    assigned = set()
    patches = []
    for seed in order:
        if seed in assigned:
            continue
        sn = normals[seed]
        assigned.add(seed)
        members = [seed]
        total = float(area[seed])
        stack = [seed]
        while stack:
            f = stack.pop()
            for g in nbrs[f]:
                if g in assigned:
                    continue
                if float(np.dot(normals[g], sn)) < cut:
                    continue
                assigned.add(g)
                members.append(g)
                total += float(area[g])
                stack.append(g)
        patches.append((members, total))
    return patches


def _free_edge(surface: Surface, part: Part, v_edge, dir_, ux, uy, vx, vy,
               u_lo, u_hi, p):
    """Does the patch end in AIR along its edge at v = `v_edge` (`dir_` -1 low
    side, +1 high)? Probed at five points: the underside just inside the edge
    gives the height, and a point past the edge is tested for part; when most
    probes find air and nothing stands in the flush row's column, the edge is
    free (its outermost row moves out flush, see `patch_tracks`)."""
    air = part_n = 0
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        u = u_lo + (u_hi - u_lo) * t
        ix = ux * u + vx * (v_edge - dir_ * 0.3)
        iy = uy * u + vy * (v_edge - dir_ * 0.3)
        z = surface.z_at([[ix, iy]])[0]
        if np.isnan(z):
            continue
        if z < p.min_height_squat:              # rests on the bed: no lip to curl
            part_n += 1
            continue
        ox = ux * u + vx * (v_edge + dir_ * 0.8)
        oy = uy * u + vy * (v_edge + dir_ * 0.8)
        solid = any(part.inside([[ox, oy, z + dz]])[0]
                    for dz in (-2.0, -1.0, -0.3, 0.3, 1.0, 2.0))
        if solid:
            part_n += 1
            continue
        for side in (1, -1):
            off = v_edge - dir_ * p.edge_inset + side * dir_ * (p.thickness / 2 + p.side_clear)
            cx = ux * u + vx * off
            cy = uy * u + vy * off
            zz = z - 0.5
            while zz > 0.3:
                if part.inside([[cx, cy, zz]])[0]:
                    return False
                zz -= 1.0
        air += 1
    return air >= 2 and air > part_n


def patch_tracks(pts, surface: Surface, step: float, span: float, p,
                 part: Part | None = None, axis=None):
    """Straight parallel tracks across one locally-flat sub-patch, sampled on the
    patch's own surface, spaced `span` apart, split wherever the patch does not
    cover them (a pocket, a hole). A track split into pieces keeps only pieces of
    at least `min_stations` -- the hole is load-bearing: a bowl's ring only covers
    short chords, and short chords are stubs."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return []
    c = pts.mean(axis=0)
    d = pts - c
    sxx = float((d[:, 0] * d[:, 0]).sum())
    sxy = float((d[:, 0] * d[:, 1]).sum())
    syy = float((d[:, 1] * d[:, 1]).sum())
    sxz = float((d[:, 0] * d[:, 2]).sum())
    syz = float((d[:, 1] * d[:, 2]).sum())
    det = sxx * syy - sxy * sxy
    gx, gy = ((syy * sxz - sxy * syz) / det, (sxx * syz - sxy * sxz) / det) \
        if det > 1e-9 else (0.0, 0.0)
    slope = math.hypot(gx, gy)

    # Run direction: driven by the underside's SLOPE, not the footprint's
    # covariance principal axis (which leans to a spurious diagonal on a messy
    # overhang). Near-flat patches align to the longer world axis.
    if axis is not None:
        ux, uy = axis
    elif slope >= p.contour_slope_min:
        ux, uy = gx, gy
    else:
        ux, uy = (1.0, 0.0) if np.ptp(pts[:, 0]) >= np.ptp(pts[:, 1]) else (0.0, 1.0)
    un = math.hypot(ux, uy)
    ux, uy = ux / un, uy / un
    vx, vy = -uy, ux

    u = pts[:, 0] * ux + pts[:, 1] * uy
    v = pts[:, 0] * vx + pts[:, 1] * vy
    u_lo, u_hi = float(u.min()), float(u.max())
    v_lo, v_hi = float(v.min()), float(v.max())
    if u_hi - u_lo < 1e-6:
        return []
    n_st = max(2, min(400, int(math.ceil((u_hi - u_lo) / step))))

    v_ext = v_hi - v_lo
    row_span = max(1.0, span)
    n = max(1, int(round(v_ext / row_span)))
    rows = [v_lo + v_ext * (w + 0.5) / n for w in range(n)]

    # Free-edge rows only under a near-flat underside: the outermost row moves out
    # flush (edge_inset), so a lip past the last wall is not a cantilever.
    flat = slope <= math.tan(math.radians(p.edge_flat_deg))
    if part is not None and flat and v_ext > 4 * p.edge_inset:
        e = p.edge_inset
        near = 2 * e
        lo = (rows[0] - v_lo > near) and _free_edge(
            surface, part, v_lo, -1, ux, uy, vx, vy, u_lo, u_hi, p)
        hi = (v_hi - rows[-1] > near) and _free_edge(
            surface, part, v_hi, +1, ux, uy, vx, vy, u_lo, u_hi, p)
        if lo or hi:
            a = v_lo + e if lo else rows[0]
            b = v_hi - e if hi else rows[-1]
            cap = max(v_ext / n, 2 * p.max_unsupported_span)
            m = max(n, int(math.ceil((b - a) / cap - 1e-9)) + 1)
            rows = ((([a] if lo else []) + [rows[0]] + ([b] if hi else [])) if n == 1 else
                    [a + (b - a) * w / (m - 1) for w in range(m)])
            rows = sorted(rows)

    tracks = []
    for v0 in rows:
        cur = []
        for k in range(n_st + 1):
            uu = u_lo + (u_hi - u_lo) * k / n_st
            x, y = ux * uu + vx * v0, uy * uu + vy * v0
            z = surface.z_at([[x, y]])[0]
            if np.isnan(z):
                if cur:
                    tracks.append(np.array(cur))
                    cur = []
            else:
                cur.append([x, y, z])
        if cur:
            tracks.append(np.array(cur))
    return [t for t in tracks if len(t) >= p.min_stations]


def _edge_points(tris, pts, gap):
    """`pts` plus points every <= `gap` mm along every edge of `tris`. A small
    tube is often meshed as a few long strips, so its vertices sit only at the two
    ends and the per-bucket lowest point zigzags; sampling the edges fixes it."""
    out = list(pts)
    tris = np.asarray(tris)
    for face in tris:
        for e in range(3):
            a, b = face[e], face[(e + 1) % 3]
            length = float(np.linalg.norm(b - a))
            n = int(math.ceil(length / gap))
            for k in range(1, n):
                out.append(a + (b - a) * (k / n))
    return np.array(out)


def _convex_about(mesh, faces, cx, cy, ux, uy, p):
    """Is this curved band the underside of a small TUBE (a pipe, peg, chimney on
    its side)? CONVEX: seen across the line, a pipe's faces tilt away from its
    lowest line; a pocket's ceiling tilts back toward it. TWO-SIDED: a pipe's
    underside wraps the line, with band on both sides."""
    wx, wy = -uy, ux                              # across the line, in plan
    centroids = mesh.triangles[faces].mean(axis=1)
    normals = mesh.face_normals[faces]
    areas = mesh.area_faces[faces]
    away = toward = left = right = 0.0
    for k in range(len(faces)):
        d = (centroids[k, 0] - cx) * wx + (centroids[k, 1] - cy) * wy
        if abs(d) < 0.05:                         # on the line: no vote
            continue
        s = (normals[k, 0] * wx + normals[k, 1] * wy) * math.copysign(1.0, d)
        if s > 0:
            away += areas[k]
        else:
            toward += areas[k]
        if d > 0:
            left += areas[k]
        else:
            right += areas[k]
    return (away > p.tube_convex_frac * (away + toward)
            and min(left, right) > p.tube_two_sided_frac * (left + right))


def tube_line(mesh, faces, pts, surface: Surface, step: float, p):
    """The single lowest-line wall for a CURVED region -- a pipe, peg or chimney
    lying on its side, and the case this tool descends from. A tube's underside
    band curves UP away from its lowest line, so one wall there supports it; a
    flat plane has no such self-support and gets rows instead. Returns a list of
    contact polylines, or None when this region is not a tube."""
    faces = np.asarray(faces)
    normals = mesh.face_normals
    area = mesh.area_faces
    region_area = float(area[faces].sum())
    small = region_area < p.tube_min_area
    if small and region_area < p.tube_small_min_area:
        return None

    mean = (normals[faces] * area[faces, None]).sum(axis=0)
    mn = float(np.linalg.norm(mean))
    if mn < 1e-9 or region_area < 1e-9:
        return None
    mean /= mn
    cut = math.cos(math.radians(p.tube_spread_deg))
    deviant = float(area[faces][(normals[faces] @ mean) < cut].sum())
    if deviant / region_area < p.tube_curved_frac:
        return None                              # flat: rows handle it

    rough_pts = _edge_points(surface.mesh.triangles, pts, step / 2) if small else np.asarray(pts)
    rough = contact_line(rough_pts, surface, step, p.tip)
    if rough is None or straightness(rough) > p.max_wander:
        return None                              # a ring, not a tube

    # RESAMPLE the way patchTracks samples a track: fit the XY axis through the
    # rough line, walk it at `step`, ask the surface for its height; where the
    # region does not cover a station the track splits, and each piece stands on
    # its own.
    c = rough[:, :2].mean(axis=0)
    d = rough[:, :2] - c
    sxx = float((d[:, 0] * d[:, 0]).sum())
    sxy = float((d[:, 0] * d[:, 1]).sum())
    syy = float((d[:, 1] * d[:, 1]).sum())
    det = sxx * syy - sxy * sxy
    lam = (sxx + syy) / 2 + math.sqrt(max(0.0, ((sxx + syy) ** 2) / 4 - det))
    ux, uy = sxy, lam - sxx
    if math.hypot(ux, uy) < 1e-9:
        ux, uy = 1.0, 0.0
    un = math.hypot(ux, uy)
    ux, uy = ux / un, uy / un
    uu = d[:, 0] * ux + d[:, 1] * uy
    u_lo, u_hi = float(uu.min()), float(uu.max())
    if u_hi - u_lo < 1e-6:
        return None
    if small and not _convex_about(mesh, faces, c[0], c[1], ux, uy, p):
        return None                              # a pocket
    n_st = max(2, min(400, int(math.ceil((u_hi - u_lo) / step))))

    lines, cur = [], []
    for k in range(n_st + 1):
        u = u_lo + (u_hi - u_lo) * k / n_st
        x, y = c[0] + ux * u, c[1] + uy * u
        z = surface.z_at([[x, y]])[0]
        if np.isnan(z):
            if cur:
                lines.append(np.array(cur))
                cur = []
        else:
            cur.append([x, y, z])
    if cur:
        lines.append(np.array(cur))
    lines = [t for t in lines if len(t) >= p.min_stations]
    return lines or None

