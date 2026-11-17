import math

import mujoco
import numpy as np
import pytest

from rl_insert.peg_hole import PegHoleParams, driven_peg_scene, peg_hole_scene

CLEARANCES = (0.001, 0.0005, 0.0002)


def _driven(p: PegHoleParams, x0: float, tilt_deg: float = 0.0, speed: float = 0.04, seconds: float = 3.0):
    """Press the peg straight down at `speed`; return the tip position and peak linear speed."""
    m = driven_peg_scene(p).compile()
    d = mujoco.MjData(m)
    d.qpos[:5] = [x0, 0, 0.003, 0, math.radians(tilt_deg)]
    d.ctrl[:] = d.qpos[:5]
    peak = 0.0
    for i in range(int(seconds / m.opt.timestep)):
        d.ctrl[2] = 0.003 - speed * i * m.opt.timestep
        mujoco.mj_step(m, d)
        peak = max(peak, float(np.linalg.norm(d.qvel[:3])))
    return d.qpos[:3].copy(), peak


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


@pytest.mark.parametrize("c", CLEARANCES)
@pytest.mark.parametrize("case", ["aligned", "chamfer", "tilted"])
def test_pressed_peg_reaches_floor(c, case):
    p = PegHoleParams(clearance=c)
    x0, tilt = {"aligned": (0, 0), "chamfer": (c + 0.7 * p.chamfer, 0), "tilted": (0.0005, 5)}[case]
    tip, peak = _driven(p, x0, tilt)
    assert tip[2] == pytest.approx(-p.depth, abs=0.0005)
    assert peak < 0.5, f"contact injected energy: peak {peak:.2f} m/s"


@pytest.mark.parametrize("c", CLEARANCES)
@pytest.mark.parametrize("past_chamfer", [0.002, 0.003, 0.004])
def test_peg_pressed_beyond_chamfer_stays_in_or_on_the_hole(c, past_chamfer):
    """Off the chamfer the peg may snap in, jam or rest on the rim, but it must never leave the
    fixture or sink through it. (It can move fast when it snaps in: the test servo pushes a 20 g
    peg with ~20 N and has no arm inertia behind it, so speed is not checked here.)"""
    p = PegHoleParams(clearance=c)
    tip, _ = _driven(p, c + p.chamfer + past_chamfer)
    assert np.all(np.isfinite(tip))
    assert np.hypot(*tip[:2]) < p.hole_radius + p.wall, "the peg left the fixture"
    assert -p.depth - 0.0005 < tip[2] < 0.003, "the peg sank through the floor or bounced up"


def test_wall_resists_push():
    """50 N sideways, more than the arm should ever apply, moves the peg < 0.1 mm into the wall."""
    p = PegHoleParams(clearance=0.0002)
    spec = peg_hole_scene(p)
    spec.option.gravity = [0, 0, 0]
    peg = spec.body("peg")
    spec.delete(spec.joint("peg_free"))
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        peg.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis)
    peg.pos = [0, 0, -p.depth / 2]
    m = spec.compile()
    d = mujoco.MjData(m)
    for _ in range(500):
        d.xfrc_applied[m.body("peg").id, :3] = [50, 0, 0]
        mujoco.mj_step(m, d)
    assert d.qpos[0] - p.clearance < 0.0001
