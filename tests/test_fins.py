import trimesh

from support_fins import generate_fins
from support_fins.cli import main


def ledge():
    """A 10 mm post with a 40x10 mm shelf cantilevered off it at z=20."""
    post = trimesh.creation.box([10, 10, 25])
    post.apply_translation([0, 0, 12.5])
    shelf = trimesh.creation.box([40, 10, 5])
    shelf.apply_translation([25, 0, 22.5])
    return trimesh.boolean.union([post, shelf]) if trimesh.boolean.engines_available \
        else trimesh.util.concatenate([post, shelf])


def test_ledge_gets_watertight_fin():
    res = generate_fins(ledge())
    assert len(res.fins) >= 1
    for f in res.fins:
        assert f.is_watertight and f.volume > 0
        assert f.bounds[0][2] == 0.0                   # sits on the bed
        assert f.bounds[1][2] <= 20.0 - 0.2 + 1e-6     # stops `gap` below the shelf


def test_cube_needs_nothing():
    assert generate_fins(trimesh.creation.box([20, 20, 20])).fins == []


def test_cli_roundtrip(tmp_path):
    src = tmp_path / 'ledge.stl'
    ledge().export(src)
    assert main([str(src)]) == 0
    out = trimesh.load(tmp_path / 'ledge_fins.stl')
    assert len(out.faces) > len(ledge().faces)


def test_cli_rotate(tmp_path):
    src = tmp_path / 'ledge.stl'
    ledge().export(src)
    assert main([str(src), '--rotate', '0,0,0']) == 0

