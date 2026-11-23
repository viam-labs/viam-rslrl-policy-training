import math
from types import SimpleNamespace

import numpy as np
import torch

from duck_train.dance import table_at
from duck_train.dance_rewards import dance_pose, dance_pose_l1, feet_contact, phase_of, table_at_torch
from duck_train.robot import JOINTS


class Asset:
    def __init__(self, q):
        self.order = list(reversed(JOINTS))  # a joint order different from the policy's, as in mjlab
        self.data = SimpleNamespace(joint_pos=torch.as_tensor(q, dtype=torch.float32))

    def find_joints(self, names, preserve_order=False):
        ids = [self.order.index(n) for n in names]
        return (ids, names) if preserve_order else (sorted(ids), [self.order[i] for i in sorted(ids)])


class Scene(dict):
    pass


def env_with(q_asset_order, phases, found=None):
    cmd = torch.stack([torch.cos(2 * math.pi * torch.tensor(phases)), torch.sin(2 * math.pi * torch.tensor(phases)),
                       torch.zeros(len(phases))], 1)
    scene = Scene(robot=Asset(q_asset_order))
    scene.sensors = {"feet_ground_contact": SimpleNamespace(data=SimpleNamespace(found=torch.as_tensor(found)))} if found is not None else {}
    return SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_command=lambda name: cmd), device="cpu")


TABLE = np.random.default_rng(0).normal(0, 0.3, (100, 14)).astype(np.float32)  # the dtype production uses


def test_phase_of_and_table_match_numpy():
    phases = [0.0, 0.005, 0.123, 0.5, 0.999, 0.9999]  # incl. a bin edge and the wrap
    env = env_with(np.zeros((len(phases), 14)), phases)
    got = phase_of(env.command_manager.get_command("twist"))
    assert torch.allclose(got, torch.tensor(phases), atol=1e-5)
    t = table_at_torch(torch.as_tensor(TABLE), got)
    for i, p in enumerate(phases):
        assert np.allclose(t[i].numpy(), table_at(TABLE, p), atol=1e-5)


def test_pose_rewards_peak_on_reference_in_asset_order():
    phases = [0.3, 0.7]
    ref_policy = np.stack([table_at(TABLE, p) for p in phases])
    asset_order = ref_policy[:, ::-1].copy()  # Asset.order is JOINTS reversed
    env = env_with(asset_order, phases)
    assert torch.allclose(dance_pose(env, table=TABLE.tolist()), torch.ones(2), atol=1e-5)
    assert torch.allclose(dance_pose_l1(env, table=TABLE.tolist()), torch.zeros(2), atol=1e-5)
    off = env_with(asset_order + 0.3, phases)
    assert (dance_pose(off, table=TABLE.tolist()) < 0.5).all()
    assert torch.allclose(dance_pose_l1(off, table=TABLE.tolist()), torch.full((2,), -0.3), atol=1e-5)


def test_feet_contact_needs_both():
    env = env_with(np.zeros((3, 14)), [0, 0, 0], found=[[1, 1], [1, 0], [0, 0]])
    assert feet_contact(env).tolist() == [1.0, 0.0, 0.0]
