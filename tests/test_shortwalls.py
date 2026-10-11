"""Short walls, the last resort: a region nothing else served may get a wall down
to `min_span_short`, kept only while it stays stocky (at most `max_short_aspect`
times as tall as it is long).
"""
import numpy as np
import trimesh

from support_fins import FinParams
from support_fins.clearance import Part
from support_fins.contact import Surface
from support_fins.core import _boxes_overlap, _keep_clear_of, place_wall

P = FinParams()


def plate_over(z, size=40.0):
    m = trimesh.creation.box([size, size, 2.0])
    m.apply_translation([0, 0, z + 1.0])   # underside at z
    return m


def short_line(z, length):
    return np.array([[x, 0.0, z] for x in np.linspace(-length / 2, length / 2, 6)])


def test_short_mode_relaxes_the_span_floor():
    m = plate_over(10.0)
    part, surf = Part(m), Surface(m)
    line = short_line(10.0, 5.0)
    fin, reason = place_wall(line, part, surf, 0.0, P)
    assert fin is None and reason == 'stub'               # 5 mm < min_span
    fin, reason = place_wall(line, part, surf, 0.0, P, short=True)
    assert fin is not None and reason is None             # relaxed to min_span_short


def test_short_mode_does_not_promote_a_toothpick():
    m = plate_over(40.0)                                  # tall wall: no relaxed aspect
    fin, reason = place_wall(short_line(40.0, 5.0), Part(m), Surface(m), 0.0, P, short=True)
    assert fin is None and reason == 'stub'               # 39.8 mm tall / 5 mm long > 6


def test_keep_clear_of_drops_an_overlapping_wall():
    a = trimesh.creation.box([10, 2, 10])
    b = trimesh.creation.box([10, 2, 10])
    b.apply_translation([0, 0.1, 0])                      # within side_clear
    fins = [a, b]
    _keep_clear_of(fins, 1, P)
    assert fins == [a]
    assert _boxes_overlap(a.bounds, b.bounds, P.side_clear)
