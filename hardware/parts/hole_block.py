from nurb import *


@part
def hole_block(
    clearance=0.35,
    peg_width=10.0,
    hole_depth=20.0,
    mouth_chamfer=1.0,
    top_radius=27.0,
    top_height=24.0,
    base_width=90.0,
    base_thickness=6.0,
    screw_hole_width=5.5,
    screw_head_width=9.0,
    screw_spacing=72.0,
    draft=False,
):
    """The fixtured hole, matching the sim (rl_insert/peg_hole.py), on a plate that bolts down.

    clearance: radial gap between the peg and the hole wall (the sim's clearance, per side)
    peg_width: diameter of the peg this hole takes
    hole_depth: how deep the hole is below its mouth
    mouth_chamfer: the 45 degree lead-in at the mouth, matching the sim's chamfer
    top_radius: radius of the round top the hole sits in; covers the G2 pad footprint (18.4 mm)
    top_height: height of the hole's mouth above the table
    base_width: side of the square plate that bolts to the table
    base_thickness: thickness of that plate
    screw_hole_width: bolt holes through the plate (5.5 is an M5's medium clearance)
    screw_head_width: room each bolt head needs on the plate (an M5 cap head is 8.5)
    screw_spacing: distance between bolt hole centres, along each side
    """
    hole_r = peg_width / 2 + clearance
    if hole_depth > top_height - 2.0:
        reject(f"hole_depth {hole_depth} leaves under 2 mm of floor: keep it below {top_height - 2.0}",
               param="hole_depth")
    if top_radius < 18.4 + 1.0:
        reject("top_radius under 19.4 mm lets a G2 pad land beside the top instead of on it",
               param="top_radius")
    if screw_spacing / 2 - screw_head_width / 2 < top_radius + 3.0 + 1.0:
        reject("bolt heads would sit on the round top's foot: widen screw_spacing", param="screw_spacing")
    if (base_width - screw_spacing) / 2 < screw_hole_width:
        reject("under a bolt's width of plate outside the bolt holes: widen base_width",
               param="base_width")

    up = (Align.CENTER, Align.CENTER, Align.MIN)
    base = Box(base_width, base_width, base_thickness, align=up)
    top = Cylinder(top_radius, top_height, align=up)
    # 3 mm structural chamfer where the round top meets the plate (concave, so not polish's job).
    root = Pos(0, 0, base_thickness) * Cone(top_radius + 3.0, top_radius, 3.0, align=up)
    body = base + top + root
    s = screw_spacing / 2
    for x, y in ((s, s), (s, -s), (-s, s), (-s, -s)):
        body -= Pos(x, y, 0) * Cylinder(screw_hole_width / 2, base_thickness, align=up)
    mouth_z = top_height - hole_depth
    body -= Pos(0, 0, mouth_z) * Cylinder(hole_r, hole_depth, align=up)
    body -= Pos(0, 0, top_height - mouth_chamfer) * Cone(hole_r, hole_r + mouth_chamfer, mouth_chamfer, align=up)
    if draft:
        return body

    # Polish everything exposed except the bed face, the hole and its chamfer (fit), and concave edges.
    bed = body.bounding_box().min.Z
    concave = set(concave_edges(body))

    def keep(e):
        bb = e.bounding_box()
        in_hole = max(abs(bb.min.X), abs(bb.max.X), abs(bb.min.Y), abs(bb.max.Y)) <= hole_r + mouth_chamfer + 0.01
        return bb.max.Z > bed + 1e-6 and not in_hole and e not in concave

    return polish(body, body.edges().filter_by(keep), 1.0)
