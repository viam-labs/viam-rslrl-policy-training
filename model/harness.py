"""Generic Viam training-job helpers shared by the peg-insert and MicroDuck jobs."""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

logger = logging.getLogger("harness")

GL_TEST = """
import mujoco
xml = "<mujoco><worldbody><light pos='0 0 1'/><geom size='.1'/></worldbody></mujoco>"
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 32, 32)
r.update_scene(d)
assert r.render().any(), "blank frame"
try:  # best effort: OSMesa returns None here
  from OpenGL import GL
  print(GL.glGetString(GL.GL_VENDOR).decode(), "|", GL.glGetString(GL.GL_RENDERER).decode())
except Exception:
  print("unknown")
r.close()
"""


def unquote(value: str) -> str:
    """Viam passes some `--key=value` values through with the shell quotes still on."""
    return value.strip("'\"")


def select_gl() -> dict:
    """Set MUJOCO_GL to a backend that renders here: cgl on macOS, else egl, else osmesa.

    Returns what was tried; "backend" is None when nothing works (video is then disabled).
    """
    candidates = ["cgl"] if platform.system() == "Darwin" else ["egl", "osmesa"]
    info: dict = {"tried": {}, "backend": None}
    for backend in candidates:
        # mujoco sets PYOPENGL_PLATFORM to match. Test on the device training will render on.
        env = {**os.environ, "MUJOCO_GL": backend, "MUJOCO_EGL_DEVICE_ID": "0"}
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        try:
            r = subprocess.run(
                [sys.executable, "-c", GL_TEST], env=env, capture_output=True, text=True, timeout=120
            )
            ok, err = r.returncode == 0, r.stderr.strip().splitlines()[-1:] or [f"exit {r.returncode}"]
        except subprocess.TimeoutExpired:
            ok, err = False, ["timed out"]
        info["tried"][backend] = "ok" if ok else err[0]
        if ok:
            # Which GL actually rendered: Mesa's llvmpipe also answers EGL, so "egl ok" alone
            # doesn't mean the GPU is rendering.
            info["renderer"] = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else None
            info["backend"] = backend
            os.environ["MUJOCO_GL"] = backend
            break
    if info["backend"] is None:
        logger.warning("no working GL backend (%s): video disabled", info["tried"])
    return info


def device() -> str:
    import torch
    import warp as wp

    return "cuda:0" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"


class VideoUploader(threading.Thread):
    """Uploads each finished mp4 in `folder` to Viam Data. Failures are logged, never raised.

    mjlab's VideoRecorder writes a whole clip at once when it ends, so a file counts as finished
    once its size has been unchanged for one poll and it is older than the poll period.
    """

    def __init__(
        self,
        folder: Path,
        steps_per_iter: int,
        *,
        tags: list[str],
        run_name: str,
        upload_part_id: str | None,
        upload_dataset_id: str | None,
        poll_s: float = 15.0,
    ):
        super().__init__(daemon=True)
        self.folder, self.steps_per_iter, self.poll_s = folder, steps_per_iter, poll_s
        self.tags, self.run_name = tags, run_name
        self.upload_part_id, self.upload_dataset_id = upload_part_id, upload_dataset_id
        self.stop = threading.Event()
        self.lock = threading.Lock()  # one sweep at a time: the final sweep waits for an in-flight one
        self.sizes: dict[Path, int] = {}
        self.done: dict[str, str | None] = {}  # file name -> uploaded file id (None: skipped/failed)

    def run(self) -> None:
        while not self.stop.wait(self.poll_s):
            self.sweep(final=False)

    def sweep(self, final: bool) -> None:
        from model.viam_upload import upload_video

        with self.lock:
            for f in sorted(self.folder.glob("*.mp4")):
                try:
                    self._handle(f, final, upload_video)
                except Exception:
                    logger.error("video sweep failed for %s:\n%s", f.name, traceback.format_exc())

    def _handle(self, f: Path, final: bool, upload_video) -> None:
        if f.name in self.done:
            return
        size = f.stat().st_size
        stable = size > 0 and self.sizes.get(f) == size and time.time() - f.stat().st_mtime > self.poll_s
        self.sizes[f] = size
        if not (final or stable):
            return
        m = re.search(r"step-(\d+)", f.name)
        tags = [*self.tags, self.run_name] + ([f"iter-{int(m[1]) // self.steps_per_iter}"] if m else [])
        try:
            ds = [self.upload_dataset_id] if self.upload_dataset_id else None
            self.done[f.name] = upload_video(str(f), self.upload_part_id, tags, ds)
            logger.info("video %s -> %s", f.name, self.done[f.name])
        except Exception:
            self.done[f.name] = None
            logger.error("video upload failed for %s:\n%s", f.name, traceback.format_exc())


def set_visible_devices(cuda: bool) -> None:
    """run_train picks cuda:LOCAL_RANK (0) of the visible devices, "" is CPU. Keep a caller's choice."""
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" if cuda else ""


def check(results: dict, name: str, fn, out: Path | None = None) -> None:
    t0 = time.perf_counter()
    try:
        results[name] = fn()
    except Exception as e:
        results[name] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}
    results[name] = {"check_s": round(time.perf_counter() - t0, 1), **results[name]}
    logger.info("%s: %s", name, json.dumps(results[name], default=str)[:2000])
    if out is not None:  # after every check, so a crash or hang later loses nothing
        text = json.dumps(results, indent=2, default=str) + "\n"
        (out / "probe.json").write_text(text)
        print(text, flush=True)
