"""Fetch Pollen's MicroDuck walk policy (velstand) at a pinned Hugging Face revision and package it for
viam:mlmodel:onnx-cpu and duck-policy.

    uv run --with onnxruntime python scripts/fetch_walk_policy.py <duck-sim dir> [out_dir]   (default dist/microduck-walk)

Writes out_dir/policy.onnx (sha256-checked), config.json (what duck-policy reads) and LICENSE-POLICY.md,
then out_dir.tar.gz for `viam packages upload --type ml_model`. The joint limits come from the duck-sim
robot file, so pass the exported duck-sim directory.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = "pollen-robotics/microduck-policies"
REVISION = "236d1835b03ac7b45d962efbf2fe3a6940206112"
FILE = "velstand.onnx"
SHA256 = "ce894d30b3343036d499bfac1102bdaa58d59a21c1b58401ecb70fb6a2f63e0a"  # HF x-linked-etag at REVISION
DEFAULT_POSE = [0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
                0.0, 0.0873, 0.4579, 0.0049, -0.4530]  # infer_policy.py:126-141
JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]


def main(duck_sim: Path, out: Path) -> None:
    import mujoco
    import onnxruntime as ort

    out.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILE}"
    data = urllib.request.urlopen(url).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"{FILE} sha256 {digest} != pinned {SHA256}")
    (out / "policy.onnx").write_bytes(data)

    session = ort.InferenceSession(str(out / "policy.onnx"))
    meta = session.get_modelmeta().custom_metadata_map
    (inp,), (outp,) = session.get_inputs(), session.get_outputs()
    assert (inp.name, list(inp.shape), outp.name, list(outp.shape)) == ("obs", [1, 61], "actions", [1, 14])
    # The metadata keeps 3 decimals; infer_policy.py:126-141 has 4. Write infer_policy's, so both sides
    # of the parity check use the same pose, and check the metadata agrees within its rounding.
    default_pose = DEFAULT_POSE
    meta_pose = [float(v) for v in meta["default_joint_pos"].split(",")]
    assert all(abs(a - b) <= 6e-4 for a, b in zip(meta_pose, DEFAULT_POSE)), meta_pose
    joint_names = meta["joint_names"].split(",")
    assert joint_names == JOINTS, joint_names
    action_scale = float(meta["action_scale"])

    model = mujoco.MjModel.from_xml_path(str(duck_sim / "duck.xml"))
    limits = [[float(v) for v in model.joint(n).range] for n in JOINTS]

    config = {
        "schema": 1,
        "source": {"repo": REPO, "revision": REVISION, "file": FILE, "sha256": SHA256, "license": "apache-2.0"},
        "model": {"input": "obs", "output": "actions", "obs_len": 61, "action_len": 14},
        "joints": JOINTS,
        "default_pose": default_pose,
        "action_scale": action_scale,
        "joint_limits": limits,
        "control_hz": 50,
        "command": {"twist_min": [-0.4, -0.3, -1.0], "twist_max": [0.4, 0.3, 1.0]},
        "safety": {},  # action_bound and max_step_rad: filled in by scripts/duck_reference.py --stats
        "observation": "gyro(3) gravity(3) q-default_pose(14) dq(14) last_action(14) command(13): microduck_rl@0daab16 infer_policy.py:656-682",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "LICENSE-POLICY.md").write_text(
        f"policy.onnx is {FILE} from https://huggingface.co/{REPO} at revision {REVISION}, "
        "Copyright Pollen Robotics, licensed Apache-2.0.\n"
    )
    pack(out)
    print(f"wrote {out}: default_pose {default_pose}, action_scale {action_scale}")


def pack(out: Path) -> None:
    with tarfile.open(out.with_suffix(".tar.gz"), "w:gz") as tar:
        for path in sorted(out.iterdir()):
            tar.add(path, arcname=path.name)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dist/microduck-walk"))
