"""Convert UFactory's G2 STEP into the gripper meshes the sim draws (rl_insert/assets/g2/*.stl).

One-off asset step; OpenCascade isn't a project dependency, so run it in a throwaway env:

    uv run --no-project --python 3.12 --with cadquery-ocp --with numpy \
        python -I scripts/g2_step_to_stl.py "Gripper G2 3D File.STEP" rl_insert/assets/g2

Writes the body (flange frame) and each finger assembly (its sliding finger body's frame, i.e. shifted
to a closed gripper). Output is metres: STEP +y (gripper axis) -> z, STEP x (finger travel) -> y,
STEP z -> x; origin at the mount face, STEP y = -41.5 mm (fingertips end 176.3 mm out, at y = 134.8).
The real fingers ride a parallel linkage; the sim slides them, so the linkage arms translate rather
than swing. Visual only.
"""

import sys

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

MOUNT_FACE_Y = -41.5  # mm
OPENING_Y = 16.6  # mm, each pad's inner face from the centre in the STEP's pose
# Solid indices in UFactory's file (185 solids): STEP x < 0 is the sim's right finger (-y).
PARTS = {
    "g2_body": (list(range(133, 181)), 0.0),
    "g2_finger_left": (list(range(67, 133)) + [183, 184], -OPENING_Y),
    "g2_finger_right": (list(range(0, 67)) + [181, 182], OPENING_Y),
}
DEFLECTION = 0.5  # mm, tessellation chord error


def read_solids(step_path: str) -> list:
    r = STEPControl_Reader()
    assert r.ReadFile(step_path) == 1, "cannot read STEP"
    r.TransferRoots()
    solids = []
    e = TopExp_Explorer(r.OneShape(), TopAbs_SOLID)
    while e.More():
        solids.append(e.Current())
        e.Next()
    assert len(solids) == 185, f"expected UFactory's 185-solid G2 file, got {len(solids)}"
    return solids


def triangles(solids: list) -> np.ndarray:
    tris = []
    for solid in solids:
        BRepMesh_IncrementalMesh(solid, DEFLECTION, False, 0.5, True)
        f = TopExp_Explorer(solid, TopAbs_FACE)
        while f.More():
            face = TopoDS.Face(f.Current())
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, loc)
            if tri is not None:
                trsf = loc.Transformation()
                nodes = np.array([[p.X(), p.Y(), p.Z()] for p in
                                  (tri.Node(i).Transformed(trsf) for i in range(1, tri.NbNodes() + 1))])
                for i in range(1, tri.NbTriangles() + 1):
                    a, b, c = tri.Triangle(i).Get()
                    if face.Orientation() == TopAbs_REVERSED:
                        b, c = c, b
                    tris.append(nodes[[a - 1, b - 1, c - 1]])
            f.Next()
    return np.array(tris)


def to_flange_frame(tris_mm: np.ndarray, shift_y: float) -> np.ndarray:
    x, y, z = tris_mm[..., 0], tris_mm[..., 1], tris_mm[..., 2]
    return np.stack([z, x + shift_y, y - MOUNT_FACE_Y], axis=-1) / 1000.0  # cyclic: still right-handed


def write_binary_stl(path: str, tris: np.ndarray) -> None:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = n, tris
    with open(path, "wb") as fh:
        fh.write(b"G2 gripper body, flange frame, metres".ljust(80, b"\0"))
        fh.write(np.uint32(len(tris)).tobytes())
        fh.write(rec.tobytes())


if __name__ == "__main__":
    solids = read_solids(sys.argv[1])
    assert sorted(i for ids, _ in PARTS.values() for i in ids) == list(range(len(solids)))
    for name, (ids, shift) in PARTS.items():
        tris = to_flange_frame(triangles([solids[i] for i in ids]), shift)
        write_binary_stl(f"{sys.argv[2]}/{name}.stl", tris)
        lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
        print(f"{name}: {len(tris)} triangles, bounds (m) {lo.round(4)} .. {hi.round(4)}")
