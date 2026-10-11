"""The wall cross-section: a thin wall standing on a flat flange (the T).

Replaces the spike's fixed 7 mm cone foot, which splayed into a sheet under a
low overhang. The flange half-width scales with wall height and is clamped.
"""
import numpy as np
import pytest

from support_fins import FinParams
from support_fins.core import foot_for, profile, sweep, wall_sections

P = FinParams()


def test_foot_scales_and_clamps():
    assert foot_for(5.0, P) == pytest.approx(P.foot_min)     # 0.6 -> floored
    assert foot_for(20.0, P) == pytest.approx(2.4)           # 0.12 * 20
    assert foot_for(50.0, P) == pytest.approx(P.foot_max)    # capped
    assert foot_for(1000.0, P) <= (P.max_unsupported_span - 1) / 2


def test_profile_is_flange_wall_then_tip():
    half = profile(P)
    top = 20.0
    assert half(0.2, top) == pytest.approx(foot_for(top, P))   # flange
    assert half(0.59, top) == pytest.approx(foot_for(top, P))
    assert half(2.0, top) == pytest.approx(P.thickness / 2)    # wall
    assert half(top, top) == pytest.approx(P.tip / 2)          # tip
    assert half(top - P.tip_height, top) == pytest.approx(P.thickness / 2)


def test_sections_are_two_solids():
    line = np.array([[0.0, 0.0, 20.2], [10.0, 0.0, 20.2], [20.0, 0.0, 20.2]])
    wall, flange = wall_sections(line, 0.0, P)
    assert wall.shape == (3, 6, 3)
    assert flange.shape == (3, 4, 3)

    # flange spans foot either side of the centreline; wall spans `th` (and necks
    # to `tip` at the contact end)
    foot = foot_for(20.0, P)
    assert np.abs(flange[0, :, 1]).max() == pytest.approx(foot)
    assert np.abs(wall[0, :, 1]).max() == pytest.approx(P.thickness / 2)
    assert np.abs(wall[0, 2:4, 1]).max() == pytest.approx(P.tip / 2)
    assert flange[:, :, 2].max() == pytest.approx(P.base_h)

    for secs in (wall, flange):
        solid = sweep(secs)
        assert solid.is_watertight and solid.volume > 0