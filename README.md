# support-fins

Add breakaway support fins to an STL, baked into the mesh. Tip a part into its strong
orientation, run this, and print without slicer supports.

```sh
uvx support-fins part.stl                 # writes part_fins.stl
uvx support-fins part.stl -o out.stl --gap 0.25 --thickness 1.0
uvx support-fins part.stl --fins-only     # just the fins, to import as a separate body
```

The part is used in the orientation it ships in and dropped onto the bed (z=0). Every
overhang steeper than `--overhang-deg` with clear air to the bed gets a fin: a flared,
chamfered foot, a thin wall, and a necked tip that stops `--gap` below the part so it
snaps off. Overhangs above other parts of the model are skipped. Run `--help` for all
dimensions (mm).

## Python API

```python
import trimesh
from support_fins import generate_fins, FinParams

res = generate_fins(trimesh.load("part.stl"), FinParams(gap=0.25))
res.mesh.export("part_fins.stl")
```

## Credit

The fin algorithm is ported from `prototype/spike_fins.py` in
[gittrahan/support-fins](https://github.com/gittrahan/support-fins) (MIT), the engine behind
[printfins.com](https://printfins.com).
