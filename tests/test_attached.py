"""Part-attached walls: an overhang above another part of the model stands its
wall on the surface below (with a 0.2 mm bottom gap too) instead of stilting to
the plate through the part.

The bed path declines these via `build_part_attached` returning no floor; here we
check the wall it builds instead, and that an ordinary bed overhang is left to the
plate path unchanged.
"""
import numpy as np
import pytest
import trimesh

from support_fins import FinParams, generate_fins
from support_fins.clearance import Part
from support_fins.contact import Surface
from support_fins.core import build_part_attached, floor_line, sweep_between

P = FinParams()


def stacked():
    """A floor plate (top at z=2) with an overhang plate above it (underside 8)."""
    floor = trimesh.creation.box([20, 20, 2])
    floor.apply_translation([0, 0, 1.0])
    over = trimesh.creation.box([40, 8, 2])
    over.apply_translation([0, 0, 9.0])
    return trimesh.util.concatenate([floor, over])


def line_at(z, n=11, length=10.0):
    return np.array([[x, 0.0, z] for x in np.linspace(-length / 2, length / 2, n)])


def test_stands_on_the_floor_below_not_the_plate():
    m = stacked()
    fin, floored = build_part_attached(line_at(8.0), Part(m), Surface(m), 0.0, P)
    assert fin is not None and floored is None
    assert fin.is_watertight and fin.volume > 0
    # bottom rests `foot_gap` above the floor (floor top = 2), not on the plate
    assert fin.bounds[0][2] == pytest.approx(2.0 + P.foot_gap)
    assert fin.bounds[1][2] == pytest.approx(8.0 - P.gap)


def test_bed_overhang_is_left_to_the_plate_path():
    over = trimesh.creation.box([40, 8, 2])
    over.apply_translation([0, 0, 9.0])
    fin, floored = build_part_attached(line_at(8.0), Part(over), Surface(over), 0.0, P)
    assert fin is None and floored is None


def test_buried_column_is_refused():
    fill = trimesh.creation.box([30, 2, 6])
    fill.apply_translation([0, 0, 5.0])   # z 2..8, fills the column the wall would use
    m = trimesh.util.concatenate([stacked(), fill])
    fin, floored = build_part_attached(line_at(8.0), Part(m), Surface(m), 0.0, P)
    assert fin is None and floored == 'buried'


def test_floor_line_reads_the_highest_surface_below():
    m = stacked()
    floor = floor_line(line_at(8.0), Surface(m), P)
    assert all(row[2] == pytest.approx(2.0) for row in floor)


def test_sweep_between_is_one_closed_solid():
    top = line_at(8.0, n=5, length=4.0)
    floor = floor_line(top, Surface(stacked()), P)
    solid = sweep_between(top, floor, 0.0, P)
    assert solid.is_watertight and solid.volume > 0


def test_generate_fins_attaches_to_the_part():
    res = generate_fins(stacked())
    assert len(res.fins) == 1
    f = res.fins[0]
    assert f.is_watertight and f.volume > 0
    assert f.bounds[0][2] > 2.0          # stands above the plate, on the floor
