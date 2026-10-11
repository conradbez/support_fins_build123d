"""Curved bands: a pipe, peg or chimney lying on its side gets ONE wall along its
lowest line (`tube_line`), because its underside band curves up away from that
line. A flat region is left to the row path, and a bowl's ring is refused.
"""
import numpy as np
import trimesh

from support_fins import FinParams, generate_fins
from support_fins.contact import Surface
from support_fins.tracks import tube_line

P = FinParams()


def lying_cylinder(radius, height, sections=48):
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    cyl.apply_translation([0, 0, radius + 5.0])
    return cyl


def underside(cyl, zmax):
    faces = np.array([f for f in range(len(cyl.faces)) if cyl.face_normals[f][2] < -0.3])
    pts = trimesh.sample.sample_surface(cyl, 6000, seed=0)[0]
    return faces, pts[pts[:, 2] < zmax]


def test_tube_line_follows_the_axis():
    cyl = lying_cylinder(10.0, 60.0)
    faces, pts = underside(cyl, 15.0)
    lines = tube_line(cyl, faces, pts, Surface(cyl), P.station_step, P)
    assert lines is not None and len(lines) == 1
    line = lines[0]
    assert line[:, 1].std() < 1.5                     # runs along the x axis
    assert abs(line[-1, 0] - line[0, 0]) > 50.0       # a long wall, not a stub


def test_tube_line_takes_a_small_convex_tube():
    cyl = lying_cylinder(2.0, 8.0)
    faces, pts = underside(cyl, 7.0)
    lines = tube_line(cyl, faces, pts, Surface(cyl), P.station_step, P)
    assert lines is not None and len(lines) == 1      # the convex small-tube route


def test_tube_line_declines_a_flat_plate():
    plate = trimesh.creation.box([60, 40, 2])
    plate.apply_translation([0, 0, 1.0])
    bottom = np.array([f for f in range(len(plate.faces)) if plate.face_normals[f][2] < -0.9])
    pts = np.column_stack([np.zeros(100), np.zeros(100), np.zeros(100)])
    assert tube_line(plate, bottom, pts, Surface(plate), P.station_step, P) is None


def test_generate_fins_supports_a_lying_tube():
    posts = []
    for x in (-25.0, 25.0):
        post = trimesh.creation.box([4, 4, 14])
        post.apply_translation([x, 0, 7.0])
        posts.append(post)
    cyl = trimesh.creation.cylinder(radius=6, height=60, sections=48)
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    cyl.apply_translation([0, 0, 20.0])
    res = generate_fins(trimesh.util.concatenate(posts + [cyl]))
    assert len(res.fins) >= 1
