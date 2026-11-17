"""Robustness of the training entry point: package before uploads, bounded uploads, local logs."""

import asyncio
import time
from pathlib import Path

import pytest

from model import training
from rl_insert.task import viam_upload


def test_upload_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY", "k")
    monkeypatch.setenv("API_KEY_ID", "12345678-1234-5678-1234-567812345678")

    async def hang(*a, **k):
        await asyncio.sleep(60)

    monkeypatch.setattr(viam_upload.ViamClient, "create_from_dial_options", hang)
    t0 = time.time()
    with pytest.raises(TimeoutError):
        viam_upload.upload_video(str(tmp_path / "x.mp4"), "part", timeout=0.2)
    assert time.time() - t0 < 5


def test_sweep_survives_vanished_file(monkeypatch, tmp_path):
    (tmp_path / "rl-video-step-0.mp4").write_bytes(b"x")
    (tmp_path / "rl-video-step-24.mp4").write_bytes(b"x")
    args = training.parse_args(["--model_output_directory=o"])
    up = training.VideoUploader(tmp_path, 24, args, poll_s=0)
    calls = []
    monkeypatch.setattr(viam_upload, "upload_video", lambda p, *a: calls.append(Path(p).name) or "id")
    real = Path.stat

    def stat(self, *a, **k):
        if self.name == "rl-video-step-0.mp4":
            raise FileNotFoundError(self)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "stat", stat)
    up.sweep(final=True)
    assert calls == ["rl-video-step-24.mp4"]


def test_cuda_visible_devices_respected(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4")
    training.set_visible_devices(True)
    assert training.os.environ["CUDA_VISIBLE_DEVICES"] == "3,4"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    training.set_visible_devices(True)
    assert training.os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_package_written_before_final_sweep_even_if_upload_hangs(monkeypatch, tmp_path):
    import mjlab.scripts.train as mt

    events = []

    def fake_run_train(task, cfg, log_dir):
        v = Path(log_dir) / "videos" / "train"
        v.mkdir(parents=True, exist_ok=True)
        (v / "rl-video-step-0.mp4").write_bytes(b"x")
        events.append(("run_train", str(log_dir)))

    monkeypatch.setattr(mt, "run_train", fake_run_train)
    monkeypatch.setattr(training, "write_package", lambda *a, **k: events.append(("package",)))

    def slow_upload(*a, **k):
        events.append(("upload",))
        time.sleep(0.3)

    monkeypatch.setattr(viam_upload, "upload_video", slow_upload)
    monkeypatch.setattr(training, "UPLOAD_JOIN_TIMEOUT", 0.1)
    out = tmp_path / "out"
    (out / "tmp").mkdir(parents=True)
    args = training.parse_args([f"--model_output_directory={out}", "--num_envs=2", "--upload_part_id=p"])
    training.train(args, {"backend": "egl"})
    names = [e[0] for e in events]
    assert names.index("package") < names.index("upload")
    assert not events[0][1].startswith(str(out))  # logs are written locally, not under out/
    assert (out / "tmp" / "logs" / args.run_name / "videos" / "train" / "rl-video-step-0.mp4").exists()


def test_inflight_upload_not_repeated_by_final_sweep(monkeypatch, tmp_path):
    (tmp_path / "rl-video-step-0.mp4").write_bytes(b"x")
    args = training.parse_args(["--model_output_directory=o"])
    up = training.VideoUploader(tmp_path, 24, args, poll_s=0.01)
    calls = []

    def slow(p, *a):
        calls.append(Path(p).name)
        time.sleep(0.5)
        return "id"

    monkeypatch.setattr(viam_upload, "upload_video", slow)
    up.start()
    time.sleep(0.2)  # thread is now mid-upload
    up.stop.set()
    up.join(timeout=0.05)
    assert up.is_alive()
    up.sweep(final=True)
    assert calls == ["rl-video-step-0.mp4"]
