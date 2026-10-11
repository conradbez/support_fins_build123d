"""Pieces of a part: the closed shells a mesh is made of, and which of them start
in mid-air.

A mesh can hold more than one closed piece -- a multi-body export, a print-in-
place assembly, or a cut that went clean through (a bore wider than the wall
around it). A piece that neither touches the plate nor sits on another piece
stands on its supports alone, and is nearly always a modelling slip. The overhang
pass cannot see it, so it is checked here once for the readout to say out loud.

Ported from web/pieces.js in https://github.com/gittrahan/support-fins (MIT).
"""
from __future__ import annotations

import numpy as np
import trimesh

RESTS_ON = 0.3
BED_EPS = 0.35


def floating_pieces(mesh: trimesh.Trimesh, rests_on: float = RESTS_ON,
                    bed_eps: float = BED_EPS):
    """Pieces of `mesh` (seated, lowest point on the bed) that start in mid-air.

    Returns a list of `{'faces', 'lowest', 'drop'}` for each outward shell whose
    lowest point hangs over the bed or another piece by more than `rests_on`. A
    piece counts as resting when a ray straight down from its lowest point meets
    another piece within `rests_on`, or when it reaches the plate."""
    comps = mesh.split(only_watertight=False)
    if len(comps) < 2:
        return []
    lifted = [c for c in comps if c.volume > 0 and c.bounds[0][2] >= bed_eps]
    if not lifted:
        return []

    out = []
    for c in lifted:
        low = np.asarray(c.vertices)[np.argmin(np.asarray(c.vertices)[:, 2])]
        rest = trimesh.util.concatenate([o for o in comps if o is not c])
        locs, _, _ = rest.ray.intersects_location(
            np.array([low]), np.array([[0.0, 0.0, -1.0]]), multiple_hits=True)
        locs = np.asarray(locs)
        if locs.ndim == 1:
            locs = locs.reshape(-1, 3)
        zs = locs[:, 2][locs[:, 2] <= low[2] + 1e-6] if len(locs) else np.empty(0)
        below = float(zs.max()) if len(zs) else -np.inf
        drop = float(low[2]) if below == -np.inf else float(low[2] - below)
        if drop > rests_on:
            out.append({'faces': len(c.faces), 'lowest': low, 'drop': drop})
    return out
