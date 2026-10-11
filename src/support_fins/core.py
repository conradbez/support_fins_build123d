"""Breakaway support-fin generation for STL meshes.

Started as a port of prototype/spike_fins.py in
https://github.com/gittrahan/support-fins (MIT, (c) 2026 gittrahan), with the
correctness fixes from the later JS engine (web/prop.js, web/prop/*) brought back:
the wall's top is fitted to the surface directly above it and settled to exactly
`gap`, curved regions are refused, and every station and finished wall is
measured against the whole part, trimmed where it would weld, or dropped.

For every downward-facing overhang region with clear air down to the bed,
extract the contact polyline along its lowest surface and sweep a fin profile
(flat flanged foot -> thin wall -> necked tip) along it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import trimesh

from .clearance import Part, longest_run, solid_clearance, stations_certified, stations_clear
from .contact import (Surface, across, contact_line, contour_top, lower_sag,
                      settle_top, straightness)
from .tracks import patch_tracks, split_region, tube_line

# A face at EXACTLY 45deg is self-supporting; real parts have many faces sitting
# on that boundary, so give the comparison ~0.008deg of slack instead of float noise.
ANGLE_EPS = 1e-4
RAYS_PER_REGION = 24
WELD_RETRIES = 4
# How far below an overhang a settle pass may still pull its top down when
# looking for a part-attached support: comfortably covers an overhang's own slope
# over a wall's length, and stays well under the smallest floor-to-overhang gap
# worth supporting, so the top settles on the overhang, never its floor.
PART_BAND = 3.0
# The tilt a lifted part-attached bottom may reach across the wall, and the most
# a corner may hang over its floor (see `_side_floors`).
MAX_DROP = 2.75
MAX_HANG = 1.0


@dataclass
class FinParams:
    thickness: float = 1.0        # wall thickness (mm): two 0.5mm passes
    base_h: float = 0.6           # height of the flat base flange (the T's foot)
    foot_min: float = 1.6         # narrowest the flange half-width ever gets (mm)
    foot_max: float = 3.0         # widest the flange half-width ever gets (mm); kept
                                  # under half the wall-to-wall pitch so a row of
                                  # flanges never fuses into one slab
    foot_ratio: float = 0.12      # flange half-width as a fraction of wall height
    max_unsupported_span: float = 12.0  # mm between walls across a wide overhang
    gap: float = 0.2              # breakaway gap below the part (mm)
    foot_gap: float = 0.2         # bottom gap for a wall standing ON the part (mm)
    tip: float = 0.6              # width of the necked tip (mm)
    tip_height: float = 1.5       # height of the neck taper (mm)
    overhang_deg: float = 45.0    # faces steeper than this from vertical need support
    min_region_area: float = 12.0  # ignore overhang slivers smaller than this (mm^2)
    min_span: float = 7.0         # a wall shorter than this isn't worth the plate space (mm)
    min_span_part: float = 3.0    # ...but a wall standing on the part costs no plate space
    # ...and a LAST-RESORT wall for a region nothing else served may be shorter,
    # so long as it stays stocky: height at most `max_short_aspect` x its length.
    min_span_short: float = 4.0
    max_short_aspect: float = 6.0
    min_height: float = 1.5       # nor is one lower than this (mm)
    side_clear: float = 0.35      # the part must stay this far off a wall's flanks (mm)
    max_wander: float = 0.10      # contact line RMS deviation / chord; above it, no wall
    station_step: float = 1.0     # mm between cross-sections
    min_stations: int = 3         # a run shorter than this is not a wall
    split_agree_deg: float = 15.0  # faces within this of a sub-patch seed's normal join it
    contour_slope_min: float = 0.15  # rise/run above which walls run down the slope
    edge_inset: float = 0.6       # inset of the outermost row from a free edge (mm)
    edge_flat_deg: float = 30.0   # free-edge inset only under undersides this flat
    # A region is CURVED (one wall under its lowest line, the tube case) when at
    # least `tube_curved_frac` of its AREA has a normal more than `tube_spread_deg`
    # from the area-weighted mean; otherwise it is a plane and gets rows of walls.
    # Below `tube_min_area` only a CONVEX tube may take the curved route.
    tube_spread_deg: float = 25.0
    tube_curved_frac: float = 0.4
    tube_min_area: float = 300.0
    tube_small_min_area: float = 12.0
    tube_convex_frac: float = 0.7
    tube_two_sided_frac: float = 0.25
    min_span_tube: float = 3.0    # a small tube's one wall is its only support
    bed_eps: float = 0.35         # faces this close to the plate are the bottom (mm)
    # SQUAT BED SUPPORT: a flanged T-wall needs ~min_height of headroom to exist
    # (gap + flange + a sliver of taper), so a bed overhang lower than that would
    # get nothing. Below `min_height` but above `min_height_squat`, build a thin
    # wall on a thin WIDE brim instead (see `sweep_squat`); below the floor the
    # overhang sits ~on the bed and self-supports, so nothing is built.
    min_height_squat: float = 0.6
    min_span_squat: float = 4.0   # squat walls are cheap; a shorter low ledge still earns one
    squat_brim_h: float = 0.4     # ~2 layers
    squat_brim_w: float = 2.5     # brim half-width; a 5 mm-wide strip along the wall


@dataclass
class FinResult:
    mesh: trimesh.Trimesh                     # part + fins
    fins: list[trimesh.Trimesh] = field(default_factory=list)
    regions: int = 0
    skipped: dict[str, int] = field(default_factory=dict)


def region_points(mesh, faces, n=1500):
    """Densify a region by sampling its surface, not its vertices: a flat ledge is
    often just 2 triangles / 4 verts, which would starve the polyline."""
    sub = mesh.submesh([faces], append=True)
    n = int(max(n, min(20000, sub.area * 4)))   # enough to fill 1 mm buckets
    try:
        pts, _ = trimesh.sample.sample_surface(sub, n, seed=0)
        return np.vstack([np.asarray(sub.vertices), np.asarray(pts)])
    except Exception:
        return np.asarray(sub.vertices)


def foot_for(h: float, p: FinParams) -> float:
    """Flange half-width for a wall `h` mm tall.

    The spike used a fixed 7mm foot, which assumes the overhang sits well above
    the plate. Real overhangs are low, so a tall fixed foot degenerated into a
    splayed sheet. Scale with the wall height instead, floored and capped. The
    cap matters: walls are laid in rows a `max_unsupported_span` pitch apart, so
    a flange wider than half that would fuse into its neighbour -- keep ~1mm of
    air between them."""
    span_cap = max(p.foot_min, (p.max_unsupported_span - 1) / 2)
    return max(p.foot_min, min(p.foot_max, span_cap, h * p.foot_ratio))


def profile(p: FinParams):
    """Half-width of the cross-section at height z for a wall whose (absolute)
    contact tip is at `top`: a flat base flange, a straight thin wall, then the
    breakaway tip taper. One definition, shared by the sweep and the clearance
    probes (they used to carry separate copies, which is how a shape change
    silently disagrees with the checker meant to catch it)."""
    def half(z, top):
        ztip = max(top - p.tip_height, p.base_h + 0.1)
        if z < p.base_h:
            return foot_for(top, p)          # flat flange (the T's foot)
        if z < ztip:
            return p.thickness / 2           # straight wall (the T's stem)
        f = (z - ztip) / max(1e-6, top - ztip)
        return p.thickness / 2 - (p.thickness - p.tip) / 2 * f   # neck to the tip

    return half


def wall_sections(line, z_bed, p: FinParams):
    """Two cross-sections per station, oriented by the local tangent: the thin
    wall (stem + tip taper) and the flat flange it stands on. They are separate
    overlapping closed solids -- the slicer unions them -- which keeps each
    section convex and sidesteps capping the T's concave outline."""
    s3 = np.column_stack([across(line), np.zeros(len(line))])
    if not np.any(s3, axis=1).all():
        return None
    wall, flange = [], []
    for pt, s in zip(line, s3):
        top = pt[2] - p.gap
        h = top - z_bed
        foot = foot_for(h, p)
        # A low station (a tail) has less headroom than the flange + tip assume,
        # so both shrink with it; identical to the tall case for h >= 1.2.
        flange_h = min(p.base_h, h / 2)
        ztip = max(top - p.tip_height, z_bed + flange_h + 0.1)
        base_top = z_bed + flange_h
        P = lambda so, z: [pt[0] + s[0] * so, pt[1] + s[1] * so, z]
        th, tip = p.thickness / 2, p.tip / 2
        wall.append([P(+th, z_bed), P(+th, ztip), P(+tip, top),
                     P(-tip, top), P(-th, ztip), P(-th, z_bed)])
        flange.append([P(+foot, z_bed), P(+foot, base_top),
                       P(-foot, base_top), P(-foot, z_bed)])
    return np.array(wall), np.array(flange)


def sweep(secs):
    """Bridge consecutive sections into a closed solid, cap both ends."""
    m, k = secs.shape[0], secs.shape[1]
    V = secs.reshape(-1, 3)
    F = []
    for i in range(m - 1):
        for j in range(k):
            a, b = i * k + j, i * k + (j + 1) % k
            c, d = (i + 1) * k + (j + 1) % k, (i + 1) * k + j
            F += [[a, b, c], [a, c, d]]
    for j in range(1, k - 1):                    # end caps (fan)
        F.append([0, j + 1, j])
        F.append([(m - 1) * k, (m - 1) * k + j, (m - 1) * k + j + 1])
    w = trimesh.Trimesh(vertices=V, faces=np.array(F), process=False)
    if w.volume < 0:
        w.invert()
    return w


def _span(line):
    return float(np.linalg.norm(line[-1, :2] - line[0, :2]))


def _boxes_overlap(a, b, pad):
    """Would two axis-aligned bounds (each [min, max]) come within `pad`?"""
    return bool(a[0][0] - pad <= b[1][0] and b[0][0] <= a[1][0] + pad
                and a[0][1] - pad <= b[1][1] and b[0][1] <= a[1][1] + pad
                and a[0][2] - pad <= b[1][2] and b[0][2] <= a[1][2] + pad)


def _keep_clear_of(fins, before, p: FinParams):
    """Drop the last `len(fins) - before` fins (the last-resort pass) that come
    within `side_clear` of a wall built before it, or of an earlier kept one.
    Conservative bounding boxes; 'no prop' is fixable, a fused one is not."""
    existing = [f.bounds for f in fins[:before]]
    kept = []
    for f in fins[before:]:
        b = f.bounds
        if any(_boxes_overlap(b, o, p.side_clear) for o in existing):
            continue
        if any(_boxes_overlap(b, k.bounds, p.side_clear) for k in kept):
            continue
        kept.append(f)
    fins[before:] = kept



def squat_sections(line, z_bed, p: FinParams):
    """Two cross-sections per station for a SQUAT wall: the thin stem (th-wide,
    necking to the breakaway tip) and a thin WIDE brim it stands on. Same two
    overlapping solids as `wall_sections`, sized for a wall too low for the full
    T-flange (base_h alone would be most of the wall)."""
    s3 = np.column_stack([across(line), np.zeros(len(line))])
    if not np.any(s3, axis=1).all():
        return None
    wall, brim = [], []
    brim_top = z_bed + p.squat_brim_h
    for pt, s in zip(line, s3):
        top = pt[2] - p.gap
        if top - z_bed < p.min_height_squat:
            return None
        ztip = max(top - p.tip_height, brim_top + 0.05)
        P = lambda so, z: [pt[0] + s[0] * so, pt[1] + s[1] * so, z]
        th, tip = p.thickness / 2, p.tip / 2
        wall.append([P(+th, z_bed), P(+th, ztip), P(+tip, top),
                     P(-tip, top), P(-th, ztip), P(-th, z_bed)])
        brim.append([P(+p.squat_brim_w, z_bed), P(+p.squat_brim_w, brim_top),
                     P(-p.squat_brim_w, brim_top), P(-p.squat_brim_w, z_bed)])
    return np.array(wall), np.array(brim)


def sweep_squat(line, z_bed, p: FinParams):
    """A squat wall as its two overlapping solids, or None if a station is too low."""
    secs = squat_sections(line, z_bed, p)
    if secs is None:
        return None
    return trimesh.util.concatenate([sweep(s) for s in secs])


def build_squat_bed(line, part: Part, surface: Surface, z_bed, p: FinParams,
                    claimed=None):
    """Build brimmed squat walls on the sub-`min_height` bed stations of a
    contoured line -- the near-bed overhangs the flanged wall can't reach.

    The low band is walked into maximal runs and each is swept via `sweep_squat`,
    with the same settle/re-trim/weld guard as the plate path. `claimed`, when
    given, is a per-station mask of stations a tall wall's low tail already
    covers; those are skipped so the two never stack."""
    h = line[:, 2] - p.gap - z_bed
    clear = stations_clear(line, part, profile(p), p)
    usable = (h >= p.min_height_squat) & (h < p.min_height) & clear
    if claimed is not None:
        usable = usable & ~np.asarray(claimed, bool)
    fins = []
    k, n = 0, len(line)
    while k < n:
        if not usable[k]:
            k += 1
            continue
        j = k
        while j < n and usable[j]:
            j += 1
        raw = line[k:j].copy()
        k = j
        if len(raw) < 3 or _span(raw) < p.min_span_squat:
            continue
        # Settling can lift a station into the tall band or drop one below the
        # squat floor; keep only what is still squat-height and measurably clear.
        settle_top(raw, surface, p.gap, p.tip)
        hh = raw[:, 2] - p.gap - z_bed
        ok = ((hh >= p.min_height_squat) & (hh < p.min_height)
              & stations_certified(raw, part, profile(p), p))
        run = longest_run(ok)
        if run is None or run[1] - run[0] < 3:
            continue
        settled = raw[run[0]:run[1]]
        if _span(settled) < p.min_span_squat:
            continue
        fin = sweep_squat(settled, z_bed, p)
        if fin is None:
            continue
        if solid_clearance(fin, part, p.gap) is not None:
            continue
        fins.append(fin)
    return fins


def _side_floors(f, offs, z, w):
    """Floor under each side of a lifted bottom, so it can TILT with the part
    instead of hanging level off the highest point. Returns [z_neg, z_pos] for the
    -w and +w sides (the across-wall sign)."""
    z_n = z_p = -np.inf
    for o, fv in zip(offs, f):
        if o <= -w:
            z_n = max(z_n, fv)
        if o >= w:
            z_p = max(z_p, fv)
    # The tilt stops at MAX_DROP; a side whose floor is further down stays level.
    z_n = z if z_n < z - MAX_DROP - MAX_HANG else max(z_n, z - MAX_DROP)
    z_p = z if z_p < z - MAX_DROP - MAX_HANG else max(z_p, z - MAX_DROP)
    lift = 0.0
    for o, fv in zip(offs, f):
        if o <= -w or o >= w:
            continue
        lift = max(lift, fv - (z_n + (z_p - z_n) * (o + w) / (2 * w)))
    z_n += lift
    z_p += lift
    if z_n <= z and z_p <= z:
        return [z_n, z_p]
    # Pin the high side at z and tilt only the low one, as far as the bulge lets it.
    pin_p = z_p > z_n
    low = (z_n - lift) if pin_p else (z_p - lift)
    for o, fv in zip(offs, f):
        if o <= -w or o >= w:
            continue
        t = ((o + w) / (2 * w)) if pin_p else ((w - o) / (2 * w))
        low = max(low, (fv - z * t) / (1 - t))
    low = min(low, z)
    return [low, z] if pin_p else [z, low]


def floor_line(top_line, surface: Surface, p: FinParams, margin: float = 1.0,
               mold: bool = False):
    """The floor contour a PART-ATTACHED support stands on: for each station, the
    HIGHEST part surface strictly below the overhang, or 0 (the plate) where
    nothing intervenes. Mirror of `contour_top`: it looks DOWN across the bottom
    and takes the highest hit, so the support lands on the part instead of driving
    to z=0. `margin` keeps the overhang's own face from being read as its floor.

    With `mold`, also reads along the wall so the bottom keeps its gap from a floor
    rising along the wall too. Rows are [x, y, z] or, when foot_gap > 0, [x, y, z,
    z_neg, z_pos] carrying the two side floors for `sweep_between`."""
    g, w, t = p.foot_gap, p.thickness / 2, p.tip / 2
    offs = [-w - g, -w, -t, 0, t, w, w + g] if g > 0 else [-t, 0, t]
    bot = []
    for i in range(len(top_line)):
        a = top_line[max(0, i - 1)]
        b = top_line[min(len(top_line) - 1, i + 1)]
        rx, ry = b[0] - a[0], b[1] - a[1]
        rn = math.hypot(rx, ry) or 1.0
        sx, sy = ry / rn, -rx / rn
        ceil = top_line[i][2] - margin
        along = [-g, 0.0, g] if (mold and g > 0) else [0.0]
        ax, ay = rx / rn, ry / rn
        f0, fm = [], []
        for o in offs:
            queries, owner = [], []
            for d in along:
                queries.append([top_line[i][0] + sx * o + ax * d,
                                top_line[i][1] + sy * o + ay * d])
                owner.append(d)
            z0 = zm = 0.0
            for hits, d in zip(surface.zs_at(queries), owner):
                for zz in hits:
                    if zz >= ceil:                       # highest surface below the overhang
                        continue
                    if zz > zm:
                        zm = zz
                    if d == 0 and zz > z0:
                        z0 = zz
            f0.append(z0)
            fm.append(zm)
        z = max(f0)
        if g <= 0:
            bot.append([top_line[i][0], top_line[i][1], z])
            continue
        sf = _side_floors(fm, offs, max(fm), w)
        bot.append([top_line[i][0], top_line[i][1], z, sf[0], sf[1]])
    return bot


MOLD_STEP = 0.25
MOLD_MIN = 0.1


def mold_line(top, surface: Surface, p: FinParams):
    """MOLD a lifted wall's bottom to the part: the stations `floor_line` reads,
    plus more wherever the floor changes fast between two of them, so the bottom
    traces the part instead of cutting straight across it. New stations sit on the
    straight line between their neighbours, top included; only the bottom changes.
    Returns (top, floor) for `sweep_between`."""
    if p.foot_gap <= 0:
        return top, floor_line(top, surface, p)
    floor = floor_line(top, surface, p, mold=True)
    for _ in range(8):
        nxt = []
        added = 0
        for i in range(len(top)):
            nxt.append(top[i])
            if i + 1 == len(top):
                break
            jump = max(abs(floor[i][k] - floor[i + 1][k]) for k in (2, 3, 4))
            if jump <= MOLD_STEP:
                continue
            if math.hypot(top[i + 1][0] - top[i][0], top[i + 1][1] - top[i][1]) < 2 * MOLD_MIN:
                continue
            nxt.append([(top[i][0] + top[i + 1][0]) / 2,
                        (top[i][1] + top[i + 1][1]) / 2,
                        (top[i][2] + top[i + 1][2]) / 2])
            added += 1
        if not added:
            break
        top = np.array(nxt, dtype=float)
        floor = floor_line(top, surface, p, mold=True)
    return top, floor


def sweep_between(top_line, bot_line, z_bed, p: FinParams, min_h: float | None = None):
    """Sweep a PART-ATTACHED wall between two contours: its top stops `gap` below
    the overhang and its bottom rests ON the floor contour (or `foot_gap` above it,
    a slicer's "bottom Z distance"). The bottom tapers to `tip` width at both ends
    -- narrowest witness mark, clean snap. A station whose floor is open plate (0)
    stands on the plate with no bottom gap. Returns a solid or None."""
    min_h = p.min_height if min_h is None else min_h
    welded = p.foot_gap <= 0
    wall = []
    for i in range(len(top_line)):
        pt = top_line[i]
        a = top_line[max(0, i - 1)]
        b = top_line[min(len(top_line) - 1, i + 1)]
        rx, ry = b[0] - a[0], b[1] - a[1]
        rn = math.hypot(rx, ry)
        if rn < 1e-9:
            return None
        rx, ry = rx / rn, ry / rn
        sx, sy = ry, -rx                      # horizontal, across the wall
        top = pt[2] - p.gap
        g = p.foot_gap if bot_line[i][2] > 0 else 0.0
        bot = bot_line[i][2] + g
        if top - bot_line[i][2] < min_h:
            return None
        if welded or len(bot_line[i]) < 5:
            b_n = b_p = bot
        else:
            b_n = min(bot_line[i][3] + g, top - 0.5)
            b_p = min(bot_line[i][4] + g, top - 0.5)
        b_hi = max(b_n, b_p)
        h = top - b_hi
        taper = min(p.tip_height, (h / 2) if welded else (h - 0.1))
        z_bot_tip = (bot + taper) if welded else b_hi
        P = lambda o, z: [pt[0] + sx * o, pt[1] + sy * o, z]
        z_top_tip, z_top, w_top = top - taper, top, p.tip / 2
        th = p.thickness / 2
        if welded:
            wall.append([P(+p.tip / 2, bot), P(+th, z_bot_tip), P(+th, z_top_tip),
                         P(+w_top, z_top), P(-w_top, z_top), P(-th, z_top_tip),
                         P(-th, z_bot_tip), P(-p.tip / 2, bot)])
        else:
            wall.append([P(+th, b_p), P(+th, z_top_tip), P(+w_top, z_top),
                         P(-w_top, z_top), P(-th, z_top_tip), P(-th, b_n)])
    return sweep(np.array(wall))


def _z_mid(top, floor, k, gap):
    return (floor[k][2] + (top[k][2] - gap)) / 2


def _buried_column(top, floor, k, part: Part, p: FinParams):
    return bool(part.inside([[top[k][0], top[k][1], _z_mid(top, floor, k, p.gap)]])[0])


def build_part_attached(line, part: Part, whole: Surface, z_bed, p: FinParams):
    """Try to stand a PART-ATTACHED wall under `line`. Returns (fin, None) when
    one is built; (None, reason) when there IS a floor but no safe wall fits (the
    caller must NOT then stilt to the plate through the part); (None, None) when
    there is no floor beneath the overhang, so the plate path should have it."""
    top = np.array(line, dtype=float)
    contour_top(top, whole, p.tip, band=PART_BAND)
    lower_sag(top, whole, band=PART_BAND)
    settle_top(top, whole, p.gap, p.tip, band=PART_BAND)
    floor = floor_line(top, whole, p)

    # A real floor under a majority of stations, or this is a bed overhang.
    real = sum(1 for f in floor if f[2] > p.gap + 0.5)
    if real < max(p.min_stations, math.ceil(len(floor) * 0.5)):
        return None, None

    bad = []
    ok = []
    for k in range(len(top)):
        if floor[k][2] <= p.gap + 0.5:
            ok.append(False)
        elif (top[k][2] - p.gap) - floor[k][2] < p.min_height:
            ok.append(False); bad.append('stub')
        elif _buried_column(top, floor, k, part, p):
            ok.append(False); bad.append('buried')
        else:
            ok.append(True)
    run = longest_run(np.array(ok))
    if run is None or run[1] - run[0] < 3:
        if bad and len(bad) * 2 >= len(floor):
            counts = {w: bad.count(w) for w in set(bad)}
            return None, max(counts, key=counts.get)
        return None, 'stub'

    sub_top, sub_floor = top[run[0]:run[1]], floor[run[0]:run[1]]
    if _span(sub_top) < p.min_span_part:
        return None, 'stub'
    mold_top, mold_floor = mold_line(sub_top, whole, p)
    fin = sweep_between(mold_top, mold_floor, z_bed, p)
    if fin is None:
        return None, 'degenerate'
    return fin, None


def place_wall(line, part: Part, surface: Surface, z_bed, p: FinParams,
               min_span: float | None = None, short: bool = False):
    """Trim, settle, certify, sweep and measure one contact line.
    Returns (fin, None) or (None, reason). `min_span` overrides the plate-space
    gate (a small tube's one wall may be shorter than `min_span`); `short` is the
    last-resort pass, relaxing it further for a stocky low wall."""
    base = p.min_span if min_span is None else min_span
    if short:
        h = float(line[:, 2].max()) - p.gap - z_bed
        min_span = min(base, max(p.min_span_short, h / p.max_short_aspect))
    else:
        min_span = base
    half = profile(p)
    tall = line[:, 2] - p.gap - z_bed >= p.min_height
    clear = stations_clear(line, part, half, p)
    run = longest_run(tall & clear)
    if run is None or run[1] - run[0] < 3:
        return None, 'blocked'
    sub = line[run[0]:run[1]].copy()
    if _span(sub) < min_span:
        return None, 'stub'

    # Settle on the trimmed run only: trimming changes which part is closest.
    settle_top(sub, surface, p.gap, p.tip)
    ok = (sub[:, 2] - p.gap - z_bed >= p.min_height) & stations_certified(sub, part, half, p)

    reason = 'blocked'
    for _ in range(WELD_RETRIES):
        run2 = longest_run(ok)
        if run2 is None or run2[1] - run2[0] < 3:
            return None, reason
        settled = sub[run2[0]:run2[1]]
        if _span(settled) < min_span:
            return None, 'stub'
        secs = wall_sections(settled, z_bed, p)
        if secs is None:
            return None, 'degenerate'
        fin = trimesh.util.concatenate([sweep(s) for s in secs])
        at = solid_clearance(fin, part, p.gap)
        if at is None:
            return fin, None
        # trim the station nearest the contact (and its neighbours), try again
        k = run2[0] + int(np.argmin(np.linalg.norm(settled[:, :2] - at, axis=1)))
        ok[max(0, k - 1):k + 2] = False
        reason = 'weld'
    return None, reason


def generate_fins(mesh: trimesh.Trimesh, params: FinParams | None = None) -> FinResult:
    """Sit `mesh` on the bed (z=0) in its current orientation and add fins."""
    p = params or FinParams()
    mesh = mesh.copy()
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    z_bed = 0.0
    part = Part(mesh)

    cut = -(math.cos(math.radians(p.overhang_deg)) + ANGLE_EPS)
    nz = mesh.face_normals[:, 2]
    on_bed = mesh.triangles[:, :, 2].max(axis=1) < z_bed + p.bed_eps
    over = (nz < cut) & (~on_bed)
    idx = np.flatnonzero(over)
    adj = mesh.face_adjacency
    keep = over[adj[:, 0]] & over[adj[:, 1]]
    comps = trimesh.graph.connected_components(adj[keep], nodes=idx)
    regions = [c for c in comps if mesh.area_faces[c].sum() >= p.min_region_area]

    fins = []
    skipped = {'sliver': 0, 'buried': 0, 'noline': 0, 'wanders': 0, 'blocked': 0,
               'stub': 0, 'weld': 0, 'degenerate': 0}
    whole = Surface(mesh)

    def place_line(line, surface, min_span=None, short=False):
        """Run one contact line through the part-attached, plate and squat paths.
        Returns True when a wall was built along this line."""
        fin, floored = build_part_attached(line, part, whole, z_bed, p)
        if fin is not None:
            fins.append(fin)
            return True
        if floored is not None:
            # There IS a floor but no safe wall fits: never stilt to the plate
            # through the part. Counted, and this line is done.
            skipped[floored] = skipped.get(floored, 0) + 1
            return False
        if straightness(line) > p.max_wander:
            skipped['wanders'] += 1
            return False
        served = False
        fin, reason = place_wall(line, part, surface, z_bed, p, min_span=min_span, short=short)
        if fin is None:
            skipped[reason] += 1
        else:
            fins.append(fin)
            served = True
        # The near-bed band a full flanged wall can't reach gets brimmed squat
        # walls. `line` is untouched by place_wall (it sweeps a trimmed copy).
        squats = build_squat_bed(line, part, surface, z_bed, p)
        if squats:
            fins.extend(squats)
            served = True
        return served

    def try_lines(cands, short=False):
        served = False
        for line, surf, ms in cands:
            if place_line(line, surf, min_span=ms, short=short):
                served = True
        return served

    def patch_candidates(r, region_surface, rpts):
        """Parallel rows over the region's locally-flat sub-patches; a region the
        15-degree split shatters falls back to grid tracks over its whole
        footprint, then to its single lowest line."""
        cands = []
        for members, area in split_region(mesh, r, p.split_agree_deg):
            if area < p.min_region_area:
                skipped['sliver'] += 1
                continue
            psurf = Surface(mesh.submesh([members], append=True))
            for t in patch_tracks(region_points(mesh, members), psurf,
                                  p.station_step, p.max_unsupported_span, p, part):
                cands.append((t, psurf, None))
        if not cands:
            for t in patch_tracks(rpts, region_surface,
                                  p.station_step, p.max_unsupported_span, p, part):
                cands.append((t, region_surface, None))
        if not cands:
            line = contact_line(rpts, region_surface, p.station_step, p.tip)
            if line is not None:
                cands.append((line, region_surface, None))
        return cands

    for r in sorted(regions, key=lambda c: -mesh.area_faces[c].sum()):
        region_surface = Surface(mesh.submesh([r], append=True))
        rpts = region_points(mesh, r)
        # CURVED BAND first: a tube (a pipe, peg or chimney on its side) wants one
        # long wall along its lowest line, not rows of short ones. `tube_line`
        # declines a flat region (rows handle it) and a bowl's ring (the ring is
        # not straight). When its wall builds, the tube wins the region.
        tubes = tube_line(mesh, r, rpts, region_surface, p.station_step, p)
        tube_cands = [(t, region_surface, p.min_span_tube) for t in tubes] if tubes else []
        if tube_cands and try_lines(tube_cands):
            continue
        cands = patch_candidates(r, region_surface, rpts)
        if not try_lines(cands):
            # LAST RESORT: a stocky short wall for a region nothing else served
            # (a ring-and-strut lattice is all 4-6 mm strut undersides). Kept only
            # where it keeps `side_clear` off every wall already built.
            before = len(fins)
            try_lines(cands, short=True)
            _keep_clear_of(fins, before, p)

    out = trimesh.util.concatenate([mesh] + fins) if fins else mesh
    return FinResult(mesh=out, fins=fins, regions=len(regions), skipped=skipped)
