"""Rows of walls under wide overhangs: `split_region` cuts a region into
locally-straight sub-patches, `patch_tracks` lays straight parallel tracks across
each, splitting wherever the patch's surface has a hole.
"""
import numpy as np
import pytest
import trimesh

from support_fins import FinParams, generate_fins
from support_fins.clearance import Part
from support_fins.contact import Surface
from support_fins.tracks import patch_tracks, split_region

P = FinParams()


def test_split_region_grows_against_the_seed_normal():
    cyl = trimesh.creation.cylinder(radius=10, height=4, sections=36)
    side = np.array([f for f in range(len(cyl.faces)) if abs(cyl.face_normals[f][2]) < 0.5])
    patches = split_region(cyl, side, P.split_agree_deg)
    assert len(patches) > 1                                   # a cylinder is not one flat patch
    seen = sorted(f for members, _ in patches for f in members)
    assert seen == sorted(side.tolist())                      # every face lands in exactly one patch


def test_split_region_keeps_a_flat_face_together():
    m = trimesh.creation.box([10, 10, 2])
    bottom = np.array([f for f in range(len(m.faces)) if m.face_normals[f][2] < -0.9])
    patches = split_region(m, bottom, P.split_agree_deg)
    assert len(patches) == 1


def wide_plate():
    m = trimesh.creation.box([60, 40, 2])
    m.apply_translation([0, 0, 1.0])                          # underside at z=0
    return m


def grid_pts(m, x, y):
    xx, yy = np.meshgrid(np.linspace(-x / 2 + 1, x / 2 - 1, 30),
                         np.linspace(-y / 2 + 1, y / 2 - 1, 20))
    return np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])


def test_patch_tracks_lays_parallel_rows_across_a_wide_patch():
    m = wide_plate()
    tracks = patch_tracks(grid_pts(m, 60, 40), Surface(m), P.station_step,
                          P.max_unsupported_span, P, Part(m))
    assert len(tracks) == 3
    # every track runs the long (x) axis, at its own fixed y
    for t in tracks:
        assert abs(t[-1, 0] - t[0, 0]) == pytest.approx(58.0, abs=0.5)
        assert t[:, 1].std() == pytest.approx(0.0, abs=1e-6)
    ys = sorted(t[0, 1] for t in tracks)
    gaps = np.diff(ys)
    assert all(g > 0 for g in gaps)                           # distinct, ordered rows
    assert max(gaps) <= 2 * P.max_unsupported_span + 1e-6


def test_patch_tracks_splits_at_a_hole():
    ring = trimesh.creation.annulus(r_min=3, r_max=10, height=0.2, sections=48)
    ring.apply_translation([0, 0, 1.0])
    pts = []
    for r in np.linspace(3.2, 9.8, 20):
        for a in np.linspace(0, 2 * np.pi, 40, endpoint=False):
            pts.append([r * np.cos(a), r * np.sin(a), 1.0])
    tracks = patch_tracks(np.array(pts), Surface(ring), P.station_step,
                          P.max_unsupported_span, P, Part(ring))
    assert len(tracks) >= 2                                   # the hole is never bridged


def test_generate_fins_puts_rows_under_a_wide_overhang():
    post = trimesh.creation.box([10, 10, 20])
    post.apply_translation([0, 0, 10.0])
    plate = trimesh.creation.box([60, 40, 2])
    plate.apply_translation([0, 0, 21.0])                     # underside at z=20
    res = generate_fins(trimesh.util.concatenate([post, plate]))
    assert len(res.fins) >= 2
    ys = sorted((f.bounds[0][1] + f.bounds[1][1]) / 2 for f in res.fins)
    assert len(set(round(y, 1) for y in ys)) == len(ys)       # parallel rows, not one wall
