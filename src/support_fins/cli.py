"""support-fins CLI: read an STL, write it back with breakaway fins baked in."""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import trimesh

from .core import FinParams, generate_fins
from .pieces import floating_pieces

HELP = {
    'thickness': 'fin wall thickness',
    'base_h': 'height of the flat base flange on the bed',
    'foot_min': 'narrowest the flange half-width ever gets',
    'foot_max': 'widest the flange half-width ever gets',
    'foot_ratio': 'flange half-width as a fraction of wall height',
    'max_unsupported_span': 'mm an overhang may bridge unsupported between walls',
    'gap': 'breakaway gap between fin tip and part',
    'foot_gap': 'bottom gap for a wall standing on the part',
    'tip': 'width of the necked tip',
    'tip_height': 'height of the neck taper',
    'overhang_deg': 'overhang angle from vertical that needs support (deg)',
    'min_region_area': 'ignore overhang regions smaller than this (mm^2)',
    'min_span': 'drop walls shorter than this',
    'min_span_part': 'drop part-attached walls shorter than this',
    'min_span_short': 'a last-resort wall may be this short',
    'max_short_aspect': 'a last-resort wall may be at most this many times as tall as long',
    'min_height': 'drop wall stations lower than this',
    'side_clear': 'least clearance between the part and a wall\'s sides',
    'max_wander': 'skip curved regions: contact line RMS deviation / length above this',
    'station_step': 'spacing between wall cross-sections',
    'min_stations': 'a run shorter than this many stations is not a wall',
    'split_agree_deg': 'faces within this angle of a sub-patch seed join it (deg)',
    'contour_slope_min': 'rise/run above which walls run down the slope',
    'edge_inset': 'inset of the outermost wall row from a free edge',
    'edge_flat_deg': 'free-edge inset only under undersides this flat (deg)',
    'tube_spread_deg': 'curved region: normals this far from the mean before it is a tube (deg)',
    'tube_curved_frac': 'curved region: share of area that must deviate to be a tube',
    'tube_min_area': 'smallest region the tube route takes (mm^2)',
    'tube_small_min_area': 'smallest convex tube the tube route takes (mm^2)',
    'tube_convex_frac': 'small tube: share of off-line area that must tilt away',
    'tube_two_sided_frac': 'small tube: least share of the off-line area on either side',
    'min_span_tube': 'a small tube\'s one wall may be this short',
    'bed_eps': 'faces this close to the bed count as the bottom',
    'min_height_squat': 'near-bed overhangs below this build a brimmed squat wall',
    'min_span_squat': 'drop squat walls shorter than this',
    'squat_brim_h': 'height of a squat wall\'s brim',
    'squat_brim_w': 'half-width of a squat wall\'s brim',
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog='support-fins',
        description='Add breakaway support fins to an STL, in the orientation it ships in. '
                    'Units are mm unless noted.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('input', type=Path, help='input mesh (STL, OBJ, PLY)')
    ap.add_argument('-o', '--output', type=Path,
                    help='output file [default: <input>_fins.stl]')
    ap.add_argument('--fins-only', action='store_true',
                    help='write only the fins, not the part')
    ap.add_argument('--rotate', metavar='RX,RY,RZ',
                    help='rotate the mesh by these degrees (about X, then Y, then Z) '
                         'before generating fins')
    for f in dataclasses.fields(FinParams):
        ap.add_argument('--' + f.name.replace('_', '-'), type=type(f.default),
                        default=f.default, help=HELP[f.name])
    args = ap.parse_args(argv)

    if not args.input.exists():
        ap.error(f'{args.input}: no such file')
    out = args.output or args.input.with_name(args.input.stem + '_fins.stl')
    params = FinParams(**{f.name: getattr(args, f.name) for f in dataclasses.fields(FinParams)})

    mesh = trimesh.load(args.input, force='mesh')
    if args.rotate:
        try:
            rx, ry, rz = (float(v) for v in args.rotate.split(','))
        except ValueError:
            ap.error('--rotate wants three comma-separated degrees, e.g. --rotate 90,0,0')
        mesh.apply_transform(trimesh.transformations.euler_matrix(
            math.radians(rx), math.radians(ry), math.radians(rz), 'sxyz'))

    seated = mesh.copy()
    seated.apply_translation([0, 0, -seated.bounds[0][2]])
    for p in floating_pieces(seated):
        print(f'warning: a piece of the part ({p["faces"]} faces) starts in mid-air '
              f'at z={p["lowest"][2]:.1f}mm ({p["drop"]:.1f}mm over what is below it); '
              f'check the model for a severed body', file=sys.stderr)

    res = generate_fins(mesh, params)
    skipped = ', '.join(f'{k} {v}' for k, v in res.skipped.items() if v) or 'none'
    print(f'{args.input.name}: {res.regions} overhang regions -> {len(res.fins)} fins '
          f'(skipped: {skipped})', file=sys.stderr)
    if not res.fins:
        print('no fins generated; nothing written', file=sys.stderr)
        return 1

    target = trimesh.util.concatenate(res.fins) if args.fins_only else res.mesh
    target.export(out)
    print(f'wrote {out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
