"""Speed sweep: one 8 s constant command per case from the reset pose through duck_reference.run, reporting
the mean body-frame vx and wz over t > 3 s (net displacement, which cancels gait sway). Shows a policy's
dead zone, to choose the parity command script from.

    uv run --project <mujoco-sim worktree> --with onnxruntime python scripts/duck_speed_sweep.py \
        <microduck_rl checkout> <duck-sim dir> <policy.onnx> [--out sweep.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duck_parity
import duck_reference as ref

CASES = [(vx, 0.0) for vx in (0.1, 0.2, 0.3, 0.4, -0.2, -0.4)] \
    + [(vx, wz) for vx in (0.0, 0.2) for wz in (0.3, -0.3, 0.6, -0.6, 1.0, -1.0)]
SKIP_S = 3.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkout", type=Path)
    ap.add_argument("duck_sim", type=Path)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = []
    for vx, wz in CASES:
        ref.set_script([(8.0, vx, 0.0, wz)])
        samples = ref.run(args.checkout.resolve(), (args.duck_sim / "duck.xml").resolve(), args.onnx.resolve())["samples"]
        # segment_means skips its first SETTLE_S: shift time so that window starts at SKIP_S
        shifted = [dict(s, t=s["t"] - (SKIP_S - duck_parity.SETTLE_S)) for s in samples]
        (mvx, mwz), = duck_parity.segment_means(shifted)
        rows.append({"cmd_vx": vx, "cmd_wz": wz, "vx": mvx, "wz": mwz, "fell": duck_parity.fell(samples)})
        print(f"cmd vx {vx:+.2f} wz {wz:+.2f} -> vx {mvx:+.3f} wz {mwz:+.3f}" + (" FELL" if rows[-1]["fell"] else ""), flush=True)
    print("\n| cmd vx | cmd wz | vx | wz | fell |\n|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['cmd_vx']:+.2f} | {r['cmd_wz']:+.2f} | {r['vx']:+.3f} | {r['wz']:+.3f} | {r['fell']} |")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
