import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from duck_parity import fell, tracking  # noqa: E402
from duck_reference import DURATION, command_at  # noqa: E402


def synth():
    x = y = th = 0.0
    out = []
    for i in range(int(DURATION / 0.1) + 1):
        t = round(i * 0.1, 4)
        out.append({"t": t, "trunk_pos": [x, y, 0.12], "gravity": [0, 0, -1],
                    "trunk_quat_wxyz": [math.cos(th / 2), 0, 0, math.sin(th / 2)]})
        vx, _, wz = command_at(t + 0.05)  # constant over each interval (boundaries are on the 0.1 grid)
        if abs(wz) < 1e-9:
            x, y = x + vx * 0.1 * math.cos(th), y + vx * 0.1 * math.sin(th)
        else:
            r = vx / wz
            x += r * (math.sin(th + wz * 0.1) - math.sin(th))
            y -= r * (math.cos(th + wz * 0.1) - math.cos(th))
        th += wz * 0.1
    return out


def test_tracking_and_fell():
    s = synth()
    r = tracking(s)
    assert r["n"] > 0 and r["rmse_vx"] < 1e-3 and r["rmse_wz"] < 1e-3
    assert not fell(s)
    s[3]["gravity"] = [0.9, 0, -0.3]
    assert fell(s)


def test_segment_means_and_report(tmp_path):
    from duck_parity import SCRIPT, report, segment_means
    s = synth()
    for (_, vx, _, wz), (mvx, mwz) in zip(SCRIPT, segment_means(s)):
        assert abs(mvx - vx) < 1e-3 and abs(mwz - wz) < 1e-3
    (tmp_path / "ref.json").write_text(__import__("json").dumps({"samples": s}))
    ep = {"episode": 0, "samples": s, "slips": 0, "walker": {"ticks": 1500, "late": 0, "tick_ms_p99": 1, "run_s": 30, "state": "running"}}
    (tmp_path / "ours.json").write_text(__import__("json").dumps([ep]))
    report(tmp_path / "ref.json", tmp_path / "ours.json", tmp_path / "r.md")
    rows = [l for l in (tmp_path / "r.md").read_text().splitlines() if l.startswith("| segment")]
    assert len(rows) == 2 * len(SCRIPT) and all(l.endswith("| pass |") for l in rows)


def _run(tmp_path, n_frozen, state="running"):
    import json
    from duck_parity import report
    s = synth()
    frozen = [dict(x, trunk_pos=[0, 0, 0.12], trunk_quat_wxyz=[1, 0, 0, 0]) for x in s]  # standing still
    w = {"ticks": 1500, "late": 0, "tick_ms_p99": 1, "run_s": 30, "state": "running"}
    eps = [{"episode": i, "samples": frozen if i < n_frozen else s, "slips": 0, "walker": dict(w)} for i in range(20)]
    eps[5]["walker"]["state"] = state
    (tmp_path / "ref.json").write_text(json.dumps({"samples": s}))
    (tmp_path / "ours.json").write_text(json.dumps(eps))
    report(tmp_path / "ref.json", tmp_path / "ours.json", tmp_path / "r.md")
    return (tmp_path / "r.md").read_text()


def test_walker_ran(tmp_path):
    md = _run(tmp_path, 0)
    assert "| walker ran every episode | pass |" in md
    assert "| walker ran every episode (failed: [5]) | FAIL |" in _run(tmp_path, 0, "stopped")


def test_outliers(tmp_path):
    md = _run(tmp_path, 1)
    assert "outliers 1/20 [0] | pass |" in md  # segment 1 vx: frozen episode is an outlier but within allowance
    assert not [l for l in md.splitlines() if l.startswith(("| segment", "| walker")) and "FAIL" in l]
    assert "outliers 3/20 [0, 1, 2] | FAIL |" in _run(tmp_path, 3)
