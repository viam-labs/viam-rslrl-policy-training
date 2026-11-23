import json
import shutil
import subprocess
import tarfile
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
                       joint_limits=[[-1.0, 1.0]] * 14,
                       command={"twist_min": [-0.4, -0.3, -1.0], "twist_max": [0.4, 0.3, 1.0]}, control_hz=50,
                       source={"task": "Mjlab-Velocity-Flat-MicroDuck", "checkpoint": "model_3999.pt"})
    assert cfg["model"] == {"input": "obs", "output": "actions", "obs_len": 61, "action_len": 14}
    assert cfg["joints"] == JOINTS and cfg["safety"] == {} and cfg["control_hz"] == 50
    assert cfg["command"] == {"twist_min": [-0.4, -0.3, -1.0], "twist_max": [0.4, 0.3, 1.0]}
    json.dumps(cfg)  # serializable


def test_parse_args_task_defaults(tmp_path):
    v = parse_args([f"--model_output_directory={tmp_path}"])
    assert v.task == "velocity" and v.max_iterations == 4000
    d = parse_args([f"--model_output_directory={tmp_path}", "--task=dance", "--dance=bob-sway",
                    "--tabular_data_file=t", "--sequences_file=s", "--binary_data_file=b"])
    assert d.task == "dance" and d.dance == "bob-sway" and d.max_iterations == 2000


def test_parse_args_dance_requires_name_and_parquet(tmp_path):
    with pytest.raises(SystemExit):
        parse_args([f"--model_output_directory={tmp_path}", "--task=dance", "--tabular_data_file=t", "--sequences_file=s"])
    with pytest.raises(SystemExit):
        parse_args([f"--model_output_directory={tmp_path}", "--task=dance", "--dance=x"])


def test_build_config_phase_variant():
    cfg = build_config(joints=JOINTS, default_pose=[0.1] * 14, action_scale=1.0, joint_limits=[[-1.0, 1.0]] * 14,
                       command={"mode": "phase", "period_s": 6.0}, control_hz=50, source={"task": "Mjlab-Dance-Flat-MicroDuck"})
    assert cfg["command"] == {"mode": "phase", "period_s": 6.0} and cfg["safety"] == {}


@pytest.mark.skipif(not VELSTAND.exists(), reason="needs dist/microduck-walk (scripts/fetch_walk_policy.py)")
def test_verify_onnx_accepts_a_pollen_policy():
    meta = verify_onnx(VELSTAND, JOINTS)
    assert float(meta["action_scale"]) == 1.0


@pytest.mark.skipif(shutil.which("uvx") is None, reason="needs uvx")
def test_duck_sdist_shim():
    subprocess.run(["./build_duck.sh"], check=True, capture_output=True)
    with tarfile.open("dist/microduck-train.tar.gz") as t:
        names = t.getnames()
        shim = t.extractfile("microduck_train-0.2.0/model/training.py").read().decode()
    assert "from duck_train.train import main" in shim
    assert not any("rl_insert" in n for n in names)


def test_package_observation_is_what_duck_policy_builds():
    from duck_policy.obs import DEFAULT_LAYOUT, obs_layout
    from duck_train.package import build_config

    cfg = build_config(joints=["j"] * 14, default_pose=[0.0] * 14, action_scale=1.0, joint_limits=[[-1, 1]] * 14,
                       command={"mode": "phase", "period_s": 6.0}, control_hz=50, source={})
    assert obs_layout(cfg) == DEFAULT_LAYOUT
