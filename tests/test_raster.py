"""Grid placement: walls across a region's whole footprint, the way a slicer's
grid support stands. Used as a fallback when the patch/tube pass leaves a curved
region unserved (a bowl, whose lowest points form a ring the straightness gate
refuses).
"""
from pathlib import Path

import trimesh

from support_fins import generate_fins

FIXTURES = Path(__file__).parent / 'fixtures'


def test_grid_fallback_holds_a_curved_region():
    res = generate_fins(trimesh.load(FIXTURES / 'bowl.stl', force='mesh'))
    assert len(res.fins) >= 1


def test_grid_fallback_walls_are_clean_solids():
    res = generate_fins(trimesh.load(FIXTURES / 'bowl.stl', force='mesh'))
    for f in res.fins:
        assert f.is_watertight and f.volume > 0
