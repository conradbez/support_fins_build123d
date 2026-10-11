"""Squat bed walls: a thin wall on a wide two-layer brim for overhangs too low
for the full flanged T-wall (min_height_squat <= height < min_height).

The full wall needs ~min_height of headroom (gap + flange + a sliver of taper);
below that, a brimmed squat wall still holds the near-bed band to the plate.
"""
import numpy as np
import pytest
import trimesh

from support_fins import FinParams
from support_fins.clearance import Part
from support_fins.contact import Surface
from support_fins.core import build_squat_bed, squat_sections, sweep_squat

P = FinParams()


def plate(bottom=1.0, size=40.0):
    """A big flat plate whose underside (the overhang) sits at `bottom`."""
    m = trimesh.creation.box([size, size, 2.0])
    m.apply_translation([0, 0, bottom + 1.0])
    return m


def line_at(z, n=11, length=10.0):
    return np.array([[x, 0.0, z] for x in np.linspace(-length / 2, length / 2, n)])


def test_builds_a_squat_fin_on_a_low_band():
    part_mesh = plate(bottom=1.0)
    fins = build_squat_bed(line_at(1.0), Part(part_mesh), Surface(part_mesh), 0.0, P)
    assert len(fins) == 1
    f = fins[0]
    assert f.is_watertight and f.volume > 0
    assert f.bounds[0][2] == pytest.approx(0.0)                 # stands on the bed
    assert f.bounds[1][2] == pytest.approx(1.0 - P.gap)         # stops `gap` below it
    assert np.abs(f.bounds[:, 1]).max() == pytest.approx(P.squat_brim_w)


def test_refuses_too_high_for_squat():
    # 1.8 - gap = 1.6 >= min_height: the tall path owns this band, not squat.
    part_mesh = plate(bottom=1.8)
    assert build_squat_bed(line_at(1.8), Part(part_mesh), Surface(part_mesh), 0.0, P) == []


def test_refuses_below_the_squat_floor():
    part_mesh = plate(bottom=0.7)
    assert build_squat_bed(line_at(0.7), Part(part_mesh), Surface(part_mesh), 0.0, P) == []


def test_claimed_stations_are_skipped():
    part_mesh = plate(bottom=1.0)
    claimed = [True] * len(line_at(1.0))
    assert build_squat_bed(line_at(1.0), Part(part_mesh), Surface(part_mesh), 0.0, P,
                           claimed=claimed) == []


def test_squat_sections_are_two_solids():
    wall, brim = squat_sections(line_at(1.0, n=3, length=4.0), 0.0, P)
    assert wall.shape == (3, 6, 3)
    assert brim.shape == (3, 4, 3)
    assert np.abs(brim[0, :, 1]).max() == pytest.approx(P.squat_brim_w)
    assert brim[:, :, 2].max() == pytest.approx(P.squat_brim_h)
    solid = sweep_squat(line_at(1.0, n=3, length=4.0), 0.0, P)
    assert solid.is_watertight and solid.volume > 0
