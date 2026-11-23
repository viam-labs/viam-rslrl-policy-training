"""Package the MicroDuck ground-contact model for viam-labs:mujoco-sim as a private Viam archive.

    uv run python scripts/export_duck_scene.py <microduck_rl checkout> [out_dir]   (default dist/duck-sim)

The checkout must be at MICRODUCK_RL_COMMIT. The meshes are CC BY-NC-SA 4.0 (microduck_rl's
LICENSE-MODELS.md): the package stays private to the org, and is never bundled in a public module.

Uses robot_groundcontact.xml, not robot_walk.xml: velstand, the walk policy, was trained on it,
infer_policy.py loads it, and it has trunk and head collision.

Writes out_dir/ with robot_groundcontact.xml, its assets/ meshes, LICENSE-MODELS.md and duck.xml: a
wrapper that includes it and adds the STAND keyframe from scene.xml as its first keyframe,
the robot's home (mujoco-sim homes a robot file from its first keyframe). Also duck_pinned.xml
(with robot_pinned.xml), the posing duck: no free joint, position actuators, trunk fixed at STAND
height. Then out_dir.tar.gz, for `viam packages upload`.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

MICRODUCK_RL_COMMIT = "0daab160d3fb1c2cbaba16a4a19052f36709dd9e"
MODEL_DIR = Path("src/mjlab_microduck/robot/microduck")


def stand_keyframe(scene_xml: str) -> str:
    """scene.xml's STAND <key .../> element, verbatim. Comments are stripped first: the file
    keeps the old STAND commented out above the STAND2-tuned one (hip_pitch 0.4579, not 0.5236)."""
    live = re.sub(r"<!--.*?-->", "", scene_xml, flags=re.S)
    match = re.search(r'<key\s+name="STAND".*?/>', live, re.S)
    if match is None:
        raise SystemExit("scene.xml has no STAND keyframe; is the checkout at the pinned commit?")
    return match.group(0)


def pinned_keyframe(key: str) -> str:
    """The STAND <key> with the free joint's 7 qpos values dropped (14 joint values remain)."""
    def drop(m: re.Match) -> str:
        return 'qpos="' + " ".join(m.group(1).split()[7:]) + '"'
    return re.sub(r'qpos="([^"]*)"', drop, key, count=1)


# Posing duck only (the walk model keeps kp 0.55): stiff enough that gravity sag is < 1 mrad. kv 0.4 plus the joint
# damping is about critically damped at kp 40 under mujoco-sim's implicitfast integrator (kv >= 1 would go unstable
# under MuJoCo's default Euler at dt 0.005).
PINNED_KP, PINNED_KV, PINNED_FORCE = 40.0, 0.4, 4.0


def write_pinned(out: Path, key: str) -> None:
    """robot_pinned.xml: robot_groundcontact.xml minus the free joint, trunk fixed at STAND's pose."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(out / "robot_groundcontact.xml"))
    joint = next((j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE), None)
    if joint is None:
        raise SystemExit("robot_groundcontact.xml has no free joint; is the checkout at the pinned commit?")
    root = joint.parent  # the trunk body
    spec.delete(joint)
    vals = [float(v) for v in re.search(r'qpos="([^"]*)"', key).group(1).split()[:7]]
    root.pos, root.quat = vals[:3], vals[3:]
    for a in spec.actuators:
        a.gainprm[0], a.biasprm[1], a.biasprm[2] = PINNED_KP, -PINNED_KP, -PINNED_KV
        a.forcerange = [-PINNED_FORCE, PINNED_FORCE]
    # MuJoCo 4 syntax (position actuators become <general ...>); mujoco-sim (mujoco>=4,<5) loads it.
    (out / "robot_pinned.xml").write_text(spec.to_xml())
    (out / "duck_pinned.xml").write_text(
        '<mujoco model="microduck_pinned">\n  <include file="robot_pinned.xml"/>\n'
        f"  <keyframe>\n    {pinned_keyframe(key)}\n  </keyframe>\n</mujoco>\n"
    )


def main(checkout: Path, out: Path) -> None:
    head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    if head != MICRODUCK_RL_COMMIT:
        raise SystemExit(f"{checkout} is at {head}; check out {MICRODUCK_RL_COMMIT}")
    src = checkout / MODEL_DIR
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy(src / "robot_groundcontact.xml", out / "robot_groundcontact.xml")
    shutil.copytree(src / "assets", out / "assets", ignore=shutil.ignore_patterns("*.part"))
    shutil.copy(checkout / "LICENSE-MODELS.md", out / "LICENSE-MODELS.md")
    key = stand_keyframe((src / "scene.xml").read_text())
    (out / "duck.xml").write_text(
        '<mujoco model="microduck">\n  <include file="robot_groundcontact.xml"/>\n'
        f"  <keyframe>\n    {key}\n  </keyframe>\n</mujoco>\n"
    )
    write_pinned(out, key)
    archive = out.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(out.rglob("*")):
            tar.add(path, arcname=path.relative_to(out))
    print(f"wrote {out} and {archive}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dist/duck-sim"))
