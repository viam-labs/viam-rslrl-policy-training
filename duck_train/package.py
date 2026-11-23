"""What the MicroDuck training job writes: policy.onnx (the normalizer is part of the graph), config.json
in duck-policy's shape (as scripts/fetch_walk_policy.py writes for Pollen's velstand), checkpoint.pt and
model_metrics.json. The safety block is left empty: scripts/duck_reference.py --stats fills it after the job,
from rollouts in our reference."""

from __future__ import annotations

import json
from pathlib import Path


def build_config(*, joints, default_pose, action_scale, joint_limits, command, control_hz, source) -> dict:
    return {
        "schema": 1,
        "source": source,
        "model": {"input": "obs", "output": "actions", "obs_len": 61, "action_len": 14},
        "joints": list(joints),
        "default_pose": [float(v) for v in default_pose],
        "action_scale": float(action_scale),
        "joint_limits": [[float(lo), float(hi)] for lo, hi in joint_limits],
        "control_hz": int(control_hz),
        "command": {k: (list(map(float, v)) if isinstance(v, (list, tuple)) else v) for k, v in command.items()},
        "safety": {},
        "observation": "gyro(3) gravity(3) q-default_pose(14) dq(14) last_action(14) command(13): microduck_rl@0daab16 infer_policy.py:656-682",
    }


def verify_onnx(path: Path, joints) -> dict:
    """The contract duck-policy relies on; returns the ONNX metadata. Raises on a mismatch."""
    import onnxruntime as ort

    s = ort.InferenceSession(str(path))
    (i,), (o,) = s.get_inputs(), s.get_outputs()
    got = (i.name, list(i.shape), o.name, list(o.shape))
    if got != ("obs", [1, 61], "actions", [1, 14]):
        raise ValueError(f"{path.name}: io {got}, expected obs [1,61] -> actions [1,14]")
    meta = s.get_modelmeta().custom_metadata_map
    if meta.get("joint_names", "").split(",") != list(joints):
        raise ValueError(f"{path.name}: joint_names {meta.get('joint_names')} != the policy order")
    return meta


def scalar_metrics(log_dir: Path) -> dict:
    """The last value of every tensorboard scalar (tag names aren't hard-coded: they differ by task)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = {}
    for events in sorted(log_dir.rglob("events.out.tfevents.*")):
        acc = EventAccumulator(str(events.parent), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            last = acc.Scalars(tag)[-1]
            out[tag] = {"value": float(last.value), "step": int(last.step)}
    return out


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")
