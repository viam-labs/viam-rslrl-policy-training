from nurb import *


@part
def peg(peg_width=10.0, peg_length=60.0, draft=False):
    """The peg the G2 holds, matching the sim (rl_insert/peg_hole.py).

    peg_width: diameter, the sim's 2 x peg_radius
    peg_length: end to end; the gripper holds 20 mm and 40 mm sticks out past the pad centre
    """
    # Square ends on purpose, like the sim: no polish (see the card's Don't).
    return Cylinder(peg_width / 2, peg_length, align=(Align.CENTER, Align.CENTER, Align.MIN))
