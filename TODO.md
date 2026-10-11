# TODO: port engine fixes from the JS back to Python

`src/support_fins/core.py` is a port of `prototype/spike_fins.py`, which was last changed
2026-07-24. The JS engine in `support-fins/web/` has had about 2.5 months of fixes since.
Only the support-generation engine is listed here. UI-only features are out of scope:
Draw mode, clicking to remove a fin, sliders, orientation scoring, viewer and i18n.

Order = how much each item improves the supports for a typical STL.
Paths are relative to `support-fins/web/`.

## P0: correctness. Without these the output can't be printed (fins fuse into the part) ✅ done 2026-10-11

Ported into `contact.py`, `clearance.py` and `core.py` `place_wall`. Covered by
`tests/test_clearance.py`. Result on the fixtures: 0 fins fused into the part, but only
2 of 11 regions get a fin (angle-coupon 1/7, dome 1/1, bowl 0/1, torus 0/2). The rest
are correctly refused as curved (`wanders`) or as walls across a slope (`blocked`).
P2 recovers them.

Before (2026-10-11): every fin on angle-coupon, dome_ceiling and bowl cuts 1–6 mm into
the part, and the torus gets fins 0.1 mm tall.

- [x] **Add a regression test first.** Every fin's closest approach to the part must be
      ≥ `gap - 0.02` mm, with zero fin points inside the part. Run it on the repo fixtures
      (`tests/fixtures/*.stl`, `calibration/angle-coupon.stl`). The old fins had 77–1,978 points inside the part.
- [x] **Re-measure each wall position's height directly above it.** After bucketing, set
      each station's z to the underside directly above its own xy, not to the slice's
      lowest point. Needs a cached "surface above (x, y)" lookup.
      → `prop/contact.js` `contactLine`, `prop/surface.js` `surfaceZAt`
- [x] **Clear the whole width of the tip.** Measure at both tip edges (±tip/2 across
      the wall) as well as the centre, and take the lowest.
      → `prop/contact.js` `contourTop`
- [x] **Keep the wall below the part between positions.** Check the surface at the
      midpoint between neighbouring stations and pull both ends down if the straight
      top would poke up into it. Run up to 3 passes.
      → `prop/contact.js` `lowerSag`
- [x] **Settle the gap to exactly 0.2 mm.** Check the finished top edge every 0.25 mm
      at the tip edges and centre. Lower each station by what it lacks. Never raise one.
      → `prop/contact.js` `settleTop`
- [x] **Space stations by distance, not count.** Put one every 1.0 mm along the line,
      not a fixed 14 per fin.
      → `prop/config.js` `stationStep`, `minStations`
- [x] **Skip curved regions properly.** Use `straightness` = RMS deviation ÷ chord
      length, and skip the region when it's > 0.10. A bowl or ring otherwise gets a
      zig-zag wall through the part.
      → `prop/contact.js` `straightness`, `prop/config.js` `maxWander`
- [x] **Reject bad walls before output.** Drop a wall if:
      - it's shorter than 1.5 mm (`minHeight`)
      - the part comes within 0.35 mm of its sides (`sideClear`)
      - any of its points is inside the part
      - its path down to the plate isn't clear (`stationIsClear`; trim the run to the
        clear stations rather than dropping the whole region)
      → `prop/clearance.js`, `prop/config.js`

## P1: wall shape. Prints better and uses less plastic

- [x] **New cross-section: a thin wall on a flat flange.** 1.0 mm wall on a flat
      0.6 mm flange, emitted as two overlapping closed solids. This replaces the
      cone-shaped 7 mm foot. Foot half-width = clamp(0.12 × wall height, 1.6, 3.0).
      → `prop/sweep.js` `sweep`, `footFor`, `profileHalf`
      Done in `core.py` `foot_for` / `profile` / `wall_sections`; covered by
      `tests/test_profile.py`.
- [x] **Short low walls.** For overhangs 0.6–1.5 mm above the plate, build a thin
      wall on a 2-layer, 5 mm-wide brim instead of skipping them.
      → `prop/squat.js` `sweepSquat`, `buildSquatBed`
      Done in `core.py` `squat_sections` / `sweep_squat` / `build_squat_bed`, wired
      into `generate_fins`; covered by `tests/test_squat.py`.
- [x] **Overhangs above other parts of the model.** The spike skips these
      (`overpart`). Stand the wall on the surface below instead of the plate, with a
      0.2 mm gap at the bottom too.
      → `prop/attached.js`, `prop/mold.js` `floorLine`, `prop/sweep.js` `sweepBetween`
      Done in `core.py` `floor_line` / `mold_line` / `sweep_between` /
      `build_part_attached`, plus `Surface.zs_at` and the `band` guards on the
      contact passes; covered by `tests/test_attached.py`.

## P2: coverage. Support every overhang, not just one wall per region

- [x] **Rows of walls under wide overhangs.** *Biggest coverage win after P0:* 5 of
      angle-coupon's 7 regions are blocked because the wall runs across the slope.
      JS runs tracks DOWN the slope when rise/run ≥ 0.15 (`contourSlopeMin`). Split a region into locally flat
      sub-patches (normals within 15°). Lay parallel walls at most 12 mm apart
      (`maxUnsupportedSpan`). Inset the outer wall 0.6 mm from a free edge on
      near-flat undersides.
      → `prop/tracks.js` `splitRegion`, `patchTracks`; `prop/config.js` `edgeInset`
      Done in `tracks.py` `split_region` / `patch_tracks`, wired into `generate_fins`;
      covered by `tests/test_tracks.py`. angle-coupon goes 1 → 12 fins.
- [x] **Curved bands: one wall along the lowest line.** Covers a pipe or peg lying
      on its side (`tubeSpreadDeg`, `tubeCurvedFrac`, `tubeMinArea`, the convex
      small-tube route).
      → `prop/tracks.js` `tubeLine`, `prop/rival.js`
      Done in `tracks.py` `tube_line` (+ the curved-band-first routing in
      `generate_fins`); covered by `tests/test_tube.py`. dome_ceiling 1 → 2 fins.
- [x] **Grid placement as a fallback.** Lay walls across the whole footprint of a
      region, the way a slicer's grid support does, and keep whichever of grid or
      patch placement covers more.
      → `prop/raster.js`
      Done as the region-level grid pass in `generate_fins` (`patch_tracks` over the
      whole region when the patch/tube pass leaves it unserved); covered by
      `tests/test_raster.py`. bowl 4 fins, and angle-coupon 12 → 16 with the short
      pass below.
- [x] **Short walls as a last resort.** For regions nothing else covered, allow
      walls down to 4 mm long if height ≤ 6 × length.
      → `fins/shortwalls.js`
      Done in `core.py` `place_wall` `short=True` + the last-resort pass and
      `_keep_clear_of` overlap guard in `generate_fins`; covered by
      `tests/test_shortwalls.py`.
- [ ] **Wedges under broad downward faces no wall reached.**
      → `fins/wedges.js`
- [ ] **Strut lattices.** A region of struts joined at nodes gets a wall under each
      strut.
      → `prop/lattice.js`

## P3: hold. Keep the part from peeling or tipping

- [ ] **Tines.** One-layer horizontal nubs along the wall top that reach the part's
      surface: 0.2 mm tall (= layer height), 0.5 mm wide, every 2 mm. Packed tighter
      at the wall ends, at least 3 per wall.
      → `prop/tines.js`, `kiss.js`, `inside.js`
- [ ] **Bed pad** when the part barely touches the plate (< 60 mm² contact). A
      0.5 mm oval extending 4 mm past the part's contact.
      → `fins/seating.js`, `fins/pad.js`, `fins/config.js`
- [ ] **Sway braces** for tall, slender parts.
      → `sway.js`
- [ ] **Bracing for tall, short walls.** Ribs on walls more than 6× as tall as they
      are long, and zigzag ties between neighbouring walls.
      → `prop/brace.js`, `prop/ties.js`

## P4: nice to have

- [x] **Warn about pieces that start in mid-air** (CLI warning only).
      → `pieces.js`
      Done in `pieces.py` `floating_pieces`, wired into `cli.py`; covered by
      `tests/test_pieces.py`.
- [ ] **Full coverage pass.** A second pass of walls under overhangs still
      unsupported; a CLI flag.
      → `fins/fill.js`, `fins/coverage.js`
- [ ] **Wall cutouts to save filament.** Diamond, triangle, arch or lattice holes;
      a CLI flag.
      → `cutout.js`
- [ ] **Second-material wall tops** as a separate 3MF object, for toolchangers.
      → `prop/crest.js`
- [x] **Rotate before generating.** A `--rotate` CLI flag to apply an orientation
      first, so users don't have to pre-rotate the STL.
      Done in `cli.py` `--rotate RX,RY,RZ`; covered by `tests/test_fins.py`.
