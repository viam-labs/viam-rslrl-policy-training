import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from duck_train.dance import ARM_SLICES, build_reference, interpolate, joint_values, load_dance, table_at, validate
from duck_train.robot import DEFAULT_POSE, JOINTS, TRUNK_Z

STAND = np.array(DEFAULT_POSE)
BOB = {"name": "t", "period_s": 6.0, "keyframes": [
    {"t": 0.0, "pose": {}},
    {"t": 0.25, "pose": {"head_roll": 0.3}},
    {"t": 0.5, "pose": {}},
    {"t": 0.75, "pose": {"head_roll": -0.3}},
]}
ROLL = JOINTS.index("head_roll")


def test_hits_keyframes_and_defaults_to_stand():
    assert np.allclose(interpolate(BOB, 0.0), STAND)
    q = interpolate(BOB, 0.25)
    assert q[ROLL] == pytest.approx(0.3) and np.allclose(np.delete(q, ROLL), np.delete(STAND, ROLL))
    assert interpolate(BOB, 0.75)[ROLL] == pytest.approx(-0.3)


def test_wraps_continuously():
    a, b = interpolate(BOB, 0.999999), interpolate(BOB, 0.0)
    assert np.abs(a - b).max() < 1e-4
    assert np.allclose(interpolate(BOB, 1.25), interpolate(BOB, 0.25))


def test_smooth_between_keyframes():
    xs = np.linspace(0, 1, 401)
    roll = np.array([interpolate(BOB, x)[ROLL] for x in xs])
    assert np.abs(np.diff(roll)).max() < 0.02  # no jumps
    assert 0.0 < interpolate(BOB, 0.125)[ROLL] < 0.3


def test_clips_to_limits():
    limits = np.tile([-0.2, 0.2], (14, 1)).astype(float)
    limits[:, 0] = np.minimum(limits[:, 0], STAND - 0.01)
    limits[:, 1] = np.maximum(limits[:, 1], STAND + 0.01)
    assert interpolate(BOB, 0.25, limits)[ROLL] == pytest.approx(0.2)


@pytest.mark.parametrize("bad", [
    {"keyframes": [{"t": 0.0, "pose": {}}, {"t": 0.5, "pose": {}}, {"t": 0.4, "pose": {}}]},  # not increasing
    {"keyframes": [{"t": 0.1, "pose": {}}]},                                                   # doesn't start at 0
    {"keyframes": [{"t": 0.0, "pose": {}}, {"t": 1.0, "pose": {}}]},                          # t must be < 1
    {"keyframes": [{"t": 0.0, "pose": {"nope": 0.1}}]},                                        # unknown joint
    {"keyframes": [{"t": 0.0, "pose": {"head_roll": 0.2}}]},                                   # t=0 not STAND
    {"keyframes": [{"pose": {}}]},                                                              # missing t
    {"keyframes": [{"t": 0.0, "pose": [1]}]},                                                   # pose not a dict
    {"keyframes": [{"t": 0.0, "pose": {"head_roll": "x"}}]},                                    # non-numeric
    {"keyframes": [{"t": 0.0, "pose": {"head_roll": float("nan")}}]},                           # non-finite
    {"keyframes": [{"t": 0.0, "pose": {"head_roll": float("inf")}}]},                           # non-finite
    {"period_s": 3.0},                                                                          # period outside 4-8 s
])
def test_validate_rejects(bad):
    with pytest.raises(ValueError):
        validate({**BOB, **bad})


def test_load_roundtrip(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(BOB))
    assert load_dance(p)["name"] == "t"


def test_table_at_interpolates_between_bin_centres():
    table = np.zeros((100, 14))
    table[:, 0] = np.arange(100)
    assert table_at(table, 0.005)[0] == pytest.approx(0.0)    # bin 0 centre
    assert table_at(table, 0.010)[0] == pytest.approx(0.5)    # halfway to bin 1
    assert table_at(table, 0.999)[0] == pytest.approx(99 * 0.6, abs=1.0)  # wraps toward bin 0

from duck_train.dance import feet_world, foot_check, limits_of

DUCK_XML = Path("dist/duck-sim/duck.xml")
needs_duck = pytest.mark.skipif(not DUCK_XML.exists(), reason="run scripts/export_duck_scene.py first")


@needs_duck
def test_feet_world_at_stand_level():
    feet = feet_world(DUCK_XML, STAND, trunk_pos=(0, 0, 0.125), trunk_quat=(1, 0, 0, 0))
    assert feet.shape == (2, 3) and abs(feet[0, 2] - feet[1, 2]) < 0.005


@needs_duck
def test_foot_check_passes_head_only_dance():
    assert foot_check(BOB, DUCK_XML) == []


@needs_duck
def test_foot_check_fails_one_knee():
    knee = JOINTS.index("left_knee")
    lift = {"name": "lift", "period_s": 6.0, "keyframes": [
        {"t": 0.0, "pose": {}},
        {"t": 0.5, "pose": {"left_knee": float(STAND[knee] + 0.8), "left_hip_pitch": float(STAND[2] + 0.6)}},
    ]}
    errors = foot_check(lift, DUCK_XML)
    assert any("lifts" in e for e in errors), errors


def test_nonuniform_wrap_is_c1():
    d = {**BOB, "keyframes": [{"t": 0.0, "pose": {}}, {"t": 0.1, "pose": {"head_roll": 0.3}},
                              {"t": 0.7, "pose": {"head_roll": -0.2}}]}
    e = 1e-6
    for t0 in (0.0, 0.1, 0.7):
        l = (interpolate(d, t0)[ROLL] - interpolate(d, t0 - e)[ROLL]) / e
        r = (interpolate(d, t0 + e)[ROLL] - interpolate(d, t0)[ROLL]) / e
        assert l == pytest.approx(r, rel=1e-3, abs=1e-4)


def _two_kf(pose):
    return {"name": "x", "period_s": 6.0, "keyframes": [{"t": 0.0, "pose": {}}, {"t": 0.5, "pose": pose}]}


@needs_duck
def test_foot_check_fails_slide():
    k = [JOINTS.index("left_knee"), JOINTS.index("right_knee")]
    errors = foot_check(_two_kf({JOINTS[i]: float(STAND[i] - 0.9) for i in k}), DUCK_XML)
    assert any("slides" in e for e in errors), errors


@needs_duck
def test_foot_check_slide_is_full_range_not_drift_from_start():
    # hip yaw +-0.4 (inside the limits) plus a little roll: each side is under 2 cm from STAND but the range exceeds 2 cm
    stand = feet_world(DUCK_XML, STAND, (0, 0, TRUNK_Z), (1, 0, 0, 0))[0, :2]
    x = None
    for roll in np.arange(0.0, 0.1, 0.005):
        dance = {**BOB, "keyframes": [{"t": 0.0, "pose": {}},
                                      {"t": 0.25, "pose": {"left_hip_yaw": 0.4, "left_hip_roll": float(roll)}},
                                      {"t": 0.5, "pose": {}},
                                      {"t": 0.75, "pose": {"left_hip_yaw": -0.4, "left_hip_roll": float(-roll)}}]}
        half = np.linalg.norm(feet_world(DUCK_XML, interpolate(dance, 0.25), (0, 0, TRUNK_Z), (1, 0, 0, 0))[0, :2] - stand)
        if 0.015 < half < 0.0195:
            x = dance
            break
    assert x is not None
    errors = foot_check(x, DUCK_XML)
    assert any("slides" in e for e in errors)
    assert not any("joint limits" in e for e in errors), errors


@needs_duck
def test_foot_check_fails_joint_limit_once_per_joint():
    j = JOINTS.index("left_hip_roll")
    hi = limits_of(DUCK_XML)[j, 1]
    errors = foot_check(_two_kf({"left_hip_roll": float(hi + 0.3)}), DUCK_XML)
    lim = [e for e in errors if "joint limits" in e]
    assert len(lim) == 1 and "left_hip_roll" in lim[0], errors


@needs_duck
def test_pinned_duck_compiles_without_free_joint():
    import mujoco

    pinned = DUCK_XML.parent / "duck_pinned.xml"
    if not pinned.exists():
        pytest.skip("re-run scripts/export_duck_scene.py")
    m = mujoco.MjModel.from_xml_path(str(pinned))
    assert m.nq == 14 and m.njnt == 14 and m.nu == 14
    assert m.nkey >= 1 and m.key("STAND").qpos.shape == (14,)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("STAND").id)
    mujoco.mj_forward(m, d)
    assert d.xpos[m.body("trunk_base").id][2] == pytest.approx(0.12, abs=0.01)  # STAND's free-joint z


def _pinned_model():
    import mujoco

    pinned = DUCK_XML.parent / "duck_pinned.xml"
    if not pinned.exists():
        pytest.skip("re-run scripts/export_duck_scene.py")
    m = mujoco.MjModel.from_xml_path(str(pinned))
    m.opt.timestep = 0.005  # mujoco-sim's physics_dt overrides the model's
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # as mujoco-sim compiles the world
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("STAND").id)
    return m, d


@needs_duck
def test_pinned_duck_holds_stand_without_sag():
    import mujoco

    m, d = _pinned_model()
    d.ctrl[:] = STAND
    for _ in range(200):
        mujoco.mj_step(m, d)
    assert np.abs(d.qpos - STAND).max() < 0.003


@needs_duck
def test_pinned_duck_tracks_moving_target():
    import mujoco

    m, d = _pinned_model()
    idx = [JOINTS.index(n) for n in ("head_roll", "head_pitch", "left_hip_roll")]
    err = 0.0
    for i in range(800):
        t = i * 0.005
        ctrl = STAND.copy()
        ctrl[idx] += 0.3 * np.sin(2 * np.pi * 0.5 * t)
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)
        assert np.isfinite(d.qpos).all()
        if t >= 0.5:
            err = max(err, np.abs(d.qpos - ctrl).max())
    assert err < 0.02


@needs_duck
def test_walk_model_actuators_unchanged():
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(DUCK_XML))
    assert np.allclose(m.actuator_gainprm[:, 0], 0.55)


import importlib.util

_spec = importlib.util.spec_from_file_location("duck_choreo", Path(__file__).resolve().parent.parent / "scripts/duck_choreo.py")
choreo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(choreo)


def test_upsert_keyframe_drops_joints_at_stand_and_keeps_order():
    q = STAND.copy()
    q[ROLL] += 0.3
    q[0] += 0.005  # within 0.01 of STAND: left out
    d = choreo.upsert_keyframe(json.loads(json.dumps(BOB)), 0.6, q)
    assert [k["t"] for k in d["keyframes"]] == [0.0, 0.25, 0.5, 0.6, 0.75]
    assert d["keyframes"][3]["pose"] == {"head_roll": pytest.approx(float(q[ROLL]))}
    d = choreo.upsert_keyframe(d, 0.6, STAND)  # replace in place
    assert d["keyframes"][3] == {"t": 0.6, "pose": {}}


def test_split_and_join_arms():
    parts = choreo.split_arms(np.arange(14.0))
    assert [len(p) for p in parts.values()] == [5, 4, 5] and list(parts) == ["left-leg", "head", "right-leg"]
    assert np.allclose(choreo.join_arms(parts), np.arange(14.0))


def test_play_returns_to_stand_when_interrupted(monkeypatch):
    import asyncio

    calls = []

    class Arm:
        def __init__(self, name):
            self.name = name

        async def do_command(self, cmd):
            calls.append((self.name, cmd["values"]))
            if len([c for c in calls if c[0] == self.name]) == 3:  # third write raises (the finally's write is the fourth)
                raise RuntimeError("boom")

    monkeypatch.setattr(choreo, "_arm", lambda robot, name: Arm(name))
    with pytest.raises(RuntimeError):
        asyncio.run(choreo.play(None, BOB, None, 1))
    assert np.allclose(choreo.join_arms({n: [v for nm, v in calls[::-1] if nm == n][0] for n in choreo.ARMS}), STAND)


@needs_duck
def test_pinned_duck_settles_after_a_step():
    import mujoco

    m, d = _pinned_model()
    knee = JOINTS.index("left_knee")
    target = STAND.copy()
    target[knee] += 0.5
    d.ctrl[:] = target
    for _ in range(int(0.3 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        assert np.all(np.isfinite(d.qpos))
    q = np.array([d.qpos[m.jnt_qposadr[m.joint(n).id]] for n in JOINTS])
    assert np.abs(q - target).max() < 0.02


SAMPLE = json.loads(Path("tests/data/dance_parquet_sample.json").read_text())


def payload_like(sample_payload: str, degrees) -> str:
    """Rebuild a real payload with new angles (whatever the nesting, the angle list is the only list)."""
    def swap(o):
        if isinstance(o, list):
            return [float(v) for v in degrees]
        if isinstance(o, dict):
            return {k: swap(v) for k, v in o.items()}
        return o
    return json.dumps(swap(json.loads(sample_payload)))


def test_joint_values_reads_real_payload():
    for row in SAMPLE:
        assert len(joint_values(row["payload"])) in (4, 5)


def test_joint_values_rejects_other_shapes():
    with pytest.raises(ValueError, match="foo"):
        joint_values('{"foo": 1}')


def write_parquet(tmp_path, dance, cycles=3, lag_s=0.04, jitter=0.0, offset=0.0, tags=None, seed=0, seqs=("s1",), periods=None):
    import pandas as pd

    rng = np.random.default_rng(seed)
    start = dt.datetime(2026, 11, 21, tzinfo=dt.timezone.utc)
    period = dance["period_s"]
    rows, seq_rows = [], []
    by_arm = {r["component_name"]: r for r in SAMPLE}
    names = list(by_arm)
    for si, sid in enumerate(seqs):
        t = 0.0
        while t < cycles * period:
            q = interpolate(dance, ((t - lag_s) / period) % 1.0) + rng.normal(0, jitter, 14) + offset
            for cname in names:
                arm = next(a for a in ("left-leg", "head", "right-leg") if cname.endswith(a))
                sl = ARM_SLICES[arm]
                rows.append({"sequence_id": sid, "timestamp": start + dt.timedelta(seconds=t + rng.uniform(0, 0.004)),
                             "part_id": "p", "component_name": cname, "method_name": by_arm[cname]["method_name"],
                             "payload": payload_like(by_arm[cname]["payload"], np.degrees(q[sl]))})
            t += 0.02
        pr = periods[si] if periods else period
        seq_rows.append({"sequence_id": sid, "tags": tags or [f"dance:{dance['name']}", f"period:{pr}"],
                         "start_at": start, "end_at": start + dt.timedelta(seconds=cycles * period)})
    pd.DataFrame(rows).to_parquet(tmp_path / "tabular.parquet")
    pd.DataFrame(seq_rows).to_parquet(tmp_path / "seq.parquet")
    pd.DataFrame(columns=["sequence_id"]).to_parquet(tmp_path / "bin.parquet")
    return tmp_path / "bin.parquet", tmp_path / "tabular.parquet", tmp_path / "seq.parquet"


def test_build_reference_recovers_the_dance(tmp_path):
    period, table = build_reference(*write_parquet(tmp_path, BOB, jitter=0.002, lag_s=0.02), dance="t")
    assert period == 6.0 and table.shape == (100, 14)
    want = np.stack([interpolate(BOB, (k + 0.5) / 100) for k in range(100)])
    assert np.abs(table - want).max() < 0.01  # spec figure, with a 20 ms actuator lag (the real duck lags ~25 ms; 40 ms would give 0.012)


@pytest.mark.parametrize("tags,match", [(["dance:other", "period:6.0"], "no sequence"),
                                        (["dance:t"], "period")])
def test_build_reference_rejects(tmp_path, tags, match):
    with pytest.raises(ValueError, match=match):
        build_reference(*write_parquet(tmp_path, BOB, tags=tags), dance="t")


def test_build_reference_rejects_inconsistent_cycles(tmp_path):
    with pytest.raises(ValueError, match="spread"):
        build_reference(*write_parquet(tmp_path, BOB, jitter=0.2), dance="t")


def test_build_reference_rejects_mixed_periods(tmp_path):
    with pytest.raises(ValueError, match="mixed periods"):
        build_reference(*write_parquet(tmp_path, BOB, seqs=("s1", "s2"), periods=(6.0, 5.0)), dance="t")


def test_build_reference_checks_stand_after_spread(tmp_path):
    with pytest.raises(ValueError, match="STAND"):
        build_reference(*write_parquet(tmp_path, BOB, offset=0.2), dance="t")
