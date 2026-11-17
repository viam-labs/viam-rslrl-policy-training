"""The gripper touching the fixture ends the episode as gripper_hit (the real G2 would land on it)."""

import torch

from rl_insert.task import mdp
from rl_insert.task.checks import make_env


def test_pads_on_fixture_top_are_a_hit():
    env = make_env(num_envs=2, device="cpu")
    env.reset()
    env.sim.forward()
    env.scene.update(env.physics_dt)
    assert not mdp.gripper_hit(env).any()

    # Env 0: raise the fixture, still centred on the peg, until its top is 0.5 mm above the pad
    # bottoms (the pads overlap the wall annulus). The peg then sits 20.75 mm deep in the 20 mm bore.
    pos, rot = mdp.tcp_pose(env)
    pad_bottom = pos + rot[:, :, 2] * 0.01875  # TCP z points down, into the hole
    mouth = mdp.hole_mouth(env).clone()
    mouth[0, :2] = mdp.true_tip(env)[0, :2]
    mouth[0, 2] = pad_bottom[0, 2] + 0.0005
    ident = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, 4)
    env.scene["hole"].write_mocap_pose_to_sim(torch.cat([mouth, ident], -1))
    env.sim.forward()
    env.scene.update(env.physics_dt)
    assert mdp.gripper_hit(env).tolist() == [True, False]
