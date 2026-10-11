"""Pieces that start in mid-air: a multi-body mesh or a cut clean through, where
a whole shell neither touches the plate nor rests on another piece."""
import trimesh

from support_fins.pieces import floating_pieces


def seated(m):
    m = m.copy()
    m.apply_translation([0, 0, -m.bounds[0][2]])
    return m


def test_flags_a_stacked_piece_with_a_gap():
    a = trimesh.creation.box([10, 10, 10])
    a.apply_translation([0, 0, 5.0])
    b = trimesh.creation.box([10, 10, 10])
    b.apply_translation([0, 0, 17.0])            # gap of 2 mm above `a`
    pieces = floating_pieces(seated(trimesh.util.concatenate([a, b])))
    assert len(pieces) == 1
    assert pieces[0]['lowest'][2] == 12.0
    assert pieces[0]['drop'] == 2.0


def test_a_resting_piece_is_not_flagged():
    a = trimesh.creation.box([10, 10, 10])
    a.apply_translation([0, 0, 5.0])
    b = trimesh.creation.box([10, 10, 10])
    b.apply_translation([0, 0, 15.0])            # sitting right on `a`
    assert floating_pieces(seated(trimesh.util.concatenate([a, b]))) == []


def test_a_single_piece_is_not_flagged():
    assert floating_pieces(trimesh.creation.box([10, 10, 10])) == []
