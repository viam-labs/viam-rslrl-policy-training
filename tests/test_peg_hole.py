import math

import mujoco
import numpy as np
import pytest

from rl_insert.peg_hole import PegHoleParams, driven_peg_scene, peg_hole_scene

CLEARANCES = (0.001, 0.0005, 0.0002)


@pytest.mark.parametrize("c", CLEARANCES)
def test_bore_fits_peg_with_clearance(c):
    p = PegHoleParams(clearance=c)
    m = peg_hole_scene(p).compile()
    d = mujoco.MjData(m)
    d.qpos[:3] = [0, 0, -p.depth / 2]
    mujoco.mj_forward(m, d)
    assert d.ncon == 0, "a centred peg touches the wall"
    d.qpos[:3] = [1.5 * c, 0, -p.depth / 2]
    mujoco.mj_forward(m, d)
    assert d.ncon > 0, "a peg 1.5x the clearance off-axis doesn't touch the wall"
