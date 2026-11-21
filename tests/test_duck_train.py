import json
from pathlib import Path

import pytest

from duck_train.package import build_config, verify_onnx
from duck_train.train import parse_args

JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
VELSTAND = Path("dist/microduck-walk/policy.onnx")


def test_parse_args_strict_and_unquotes(tmp_path):
    a = parse_args([f"--model_output_directory='{tmp_path}'", "--dataset_file=x", "--mode=probe",
                    "--max_iterations='20'", "--num_envs=64", "--run_name=p1"])
    assert a.model_output_directory == tmp_path and a.mode == "probe" and a.max_iterations == 20
    with pytest.raises(SystemExit):
        parse_args([f"--model_output_directory={tmp_path}", "--nmu_envs=1"])


def test_build_config_shape():
    cfg = build_config(joints=JOINTS, default_pose=[0.1] * 14, action_scale=1.0,
                       joint_limits=[[-1.0, 1.0]] * 14, twist_min=[-0.4, -0.3, -1.0],
                       twist_max=[0.4, 0.3, 1.0], control_hz=50,
                       source={"task": "Mjlab-Velocity-Flat-MicroDuck", "checkpoint": "model_3999.pt"})
    assert cfg["model"] == {"input": "obs", "output": "actions", "obs_len": 61, "action_len": 14}
    assert cfg["joints"] == JOINTS and cfg["safety"] == {} and cfg["control_hz"] == 50
    assert cfg["command"] == {"twist_min": [-0.4, -0.3, -1.0], "twist_max": [0.4, 0.3, 1.0]}
    json.dumps(cfg)  # serializable


@pytest.mark.skipif(not VELSTAND.exists(), reason="needs dist/microduck-walk (scripts/fetch_walk_policy.py)")
def test_verify_onnx_accepts_a_pollen_policy():
    meta = verify_onnx(VELSTAND, JOINTS)
    assert float(meta["action_scale"]) == 1.0
