"""Regression: no fin may fuse into the part.

Measured independently of the generator's own weld rule:
  - no fin surface point is inside the part
  - straight up from any fin point, the part is at least `gap` away (less tolerance)
  - in 3D, nothing comes closer than the slope-adjusted floor 0.675 * gap
"""
from pathlib import Path

import numpy as np
import pytest
import trimesh

from support_fins import FinParams, generate_fins

FIXTURES = sorted((Path(__file__).parent / 'fixtures').glob('*.stl'))
GAP = FinParams().gap


def seated(path):
    m = trimesh.load(path, force='mesh')
    m.apply_translation([0, 0, -m.bounds[0][2]])
    return m


def violations(part, fin, gap=GAP):
    pts, _ = trimesh.sample.sample_surface(fin, 20000, seed=1)
    pts = np.vstack([fin.vertices, pts])
    inside = int(part.contains(pts).sum())

    hits, ri, _ = part.ray.intersects_location(
        pts, np.tile([0, 0, 1.0], (len(pts), 1)), multiple_hits=False)
    vert = float((hits[:, 2] - pts[ri, 2]).min()) if len(ri) else np.inf

    _, d, _ = trimesh.proximity.closest_point(part, pts)
    return inside, vert, float(d.min())


@pytest.mark.parametrize('path', FIXTURES, ids=[p.name for p in FIXTURES])
def test_no_fin_fuses_into_part(path):
    part = seated(path)
    res = generate_fins(trimesh.load(path, force='mesh'))
    for i, fin in enumerate(res.fins):
        inside, vert, d3 = violations(part, fin)
        assert inside == 0, f'fin {i}: {inside} points inside the part'
        assert vert >= GAP - 0.02, f'fin {i}: vertical gap {vert:.3f} mm'
        assert d3 >= 0.675 * GAP - 0.005, f'fin {i}: 3D clearance {d3:.3f} mm'


@pytest.mark.parametrize('path', FIXTURES, ids=[p.name for p in FIXTURES])
def test_fins_are_real_walls(path):
    res = generate_fins(trimesh.load(path, force='mesh'))
    for f in res.fins:
        assert f.is_watertight and f.volume > 0
        assert f.bounds[1][2] - f.bounds[0][2] >= FinParams().min_height
