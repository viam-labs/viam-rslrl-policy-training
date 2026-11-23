"""Reward math for Mjlab-Dance-Flat-MicroDuck, torch only (tested off-GPU with a fake env). The env is used
the way microduck_rl's _phase_pose_error uses it (MD tasks/mdp.py ~2917): phase from the twist command,
joints resolved by name (the asset's joint order isn't the policy's)."""

from __future__ import annotations

import torch

from .robot import JOINTS

_TABLES: dict = {}


def phase_of(cmd: torch.Tensor) -> torch.Tensor:
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0


def table_at_torch(table: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """(bins, 14) x (B,) -> (B, 14). Linear between bin centres, wrapping: same rule as dance.table_at."""
    bins = table.shape[0]
    x = phase * bins - 0.5
    i0 = torch.floor(x)
    w = (x - i0).unsqueeze(-1)
    i0 = i0.long() % bins
    return (1 - w) * table[i0] + w * table[(i0 + 1) % bins]


def _table(table, device) -> torch.Tensor:
    # ponytail: keyed by the params list's id; one table per run, so the cache never grows past one entry per device.
    key = (id(table), str(device))
    if key not in _TABLES:
        _TABLES[key] = torch.as_tensor(table, dtype=torch.float32, device=device)
    return _TABLES[key]


def _error(env, command_name, table, asset_name):
    asset = env.scene[asset_name]
    ids, _ = asset.find_joints(JOINTS, preserve_order=True)
    cur = asset.data.joint_pos[:, ids]
    tgt = table_at_torch(_table(table, cur.device), phase_of(env.command_manager.get_command(command_name)))
    return cur - tgt


def dance_pose(env, command_name: str = "twist", table=None, std: float = 0.3, asset_name: str = "robot") -> torch.Tensor:
    return torch.exp(-(_error(env, command_name, table, asset_name) / std) ** 2).mean(dim=-1)


def dance_pose_l1(env, command_name: str = "twist", table=None, asset_name: str = "robot") -> torch.Tensor:
    return -_error(env, command_name, table, asset_name).abs().mean(dim=-1)


def feet_contact(env, sensor_name: str = "feet_ground_contact") -> torch.Tensor:
    found = env.scene.sensors[sensor_name].data.found
    return (found[:, :2] > 0).all(dim=-1).float()
