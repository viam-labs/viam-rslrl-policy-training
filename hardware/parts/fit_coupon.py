from nurb import *


@part
def fit_coupon(
    peg_width=10.0,
    smallest_clearance=0.2,
    second_clearance=0.35,
    third_clearance=0.5,
    largest_clearance=1.0,
    hole_depth=10.0,
    mouth_chamfer=1.0,
    hole_spacing=18.0,
    bar_height=12.0,
    draft=False,
):
    """Four short holes at the hole_block's clearances, to try the printed peg in before printing a block.

    peg_width: diameter of the peg the holes take
    smallest_clearance: radial gap of the hole at the notched end
    second_clearance: radial gap of the next hole along
    third_clearance: radial gap of the third hole
    largest_clearance: radial gap of the hole at the plain end
    hole_depth: how deep each hole is
    mouth_chamfer: the 45 degree lead-in at each mouth, as on the hole_block
    hole_spacing: distance between hole centres, and the bar's width
    bar_height: height of the bar
    """
    clearances = (smallest_clearance, second_clearance, third_clearance, largest_clearance)
    if hole_depth > bar_height - 2.0:
        reject(f"hole_depth {hole_depth} leaves under 2 mm of floor: keep it below {bar_height - 2.0}",
               param="hole_depth")
    if hole_spacing - (peg_width + 2 * (max(clearances) + mouth_chamfer)) < 4.0:
        reject("under 4 mm of wall between neighbouring holes: widen hole_spacing", param="hole_spacing")

    up = (Align.CENTER, Align.CENTER, Align.MIN)
    length = hole_spacing * len(clearances)
    body = Box(length, hole_spacing, bar_height, align=up)
    # A 4 mm corner cut marks the smallest-clearance end (no text on parts).
    end = -length / 2
    body -= Pos(end, hole_spacing / 2, 0) * Rot(0, 0, 45) * Box(4.0 * 2**0.5, 4.0 * 2**0.5, bar_height, align=up)
    centres = [end + hole_spacing * (i + 0.5) for i in range(len(clearances))]
    for x, c in zip(centres, clearances):
        r = peg_width / 2 + c
        body -= Pos(x, 0, bar_height - hole_depth) * Cylinder(r, hole_depth, align=up)
        body -= Pos(x, 0, bar_height - mouth_chamfer) * Cone(r, r + mouth_chamfer, mouth_chamfer, align=up)
    if draft:
        return body

    bed = body.bounding_box().min.Z
    concave = set(concave_edges(body))
    reach = peg_width / 2 + max(clearances) + mouth_chamfer + 0.01

    def keep(e):
        bb = e.bounding_box()
        mid_x = (bb.min.X + bb.max.X) / 2
        in_hole = any(abs(mid_x - x) < 0.01 + 1e-3 and bb.max.X - bb.min.X <= 2 * reach for x in centres) \
            and max(abs(bb.min.Y), abs(bb.max.Y)) <= reach
        return bb.max.Z > bed + 1e-6 and not in_hole and e not in concave

    return polish(body, body.edges().filter_by(keep), 1.0)
