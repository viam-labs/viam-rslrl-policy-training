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
