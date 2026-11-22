"""Reference with +-0.01 rad uniform joint noise at t = 24 s (backward start), 20 seeds: spread of segment-5 means."""
import json, sys
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, "scripts")
import duck_reference as ref, duck_parity as par
ref.set_script(json.load(open("scripts/parity_velocity.json")))
_step = mujoco.mj_step
state = {}
def step(m, d):
    if not state["done"] and d.time >= 24.0:
        d.qpos[7:] += state["rng"].uniform(-0.01, 0.01, d.qpos.size - 7); state["done"] = True
    _step(m, d)
mujoco.mj_step = step
ck, sim, onnx = map(Path, sys.argv[1:4])
out = []
for seed in range(20):
    state.update(rng=np.random.default_rng(seed), done=False)
    r = ref.run(ck, (sim / "duck.xml").resolve(), onnx.resolve())
    vx, wz = par.segment_means(r["samples"])[5]
    out.append((seed, round(vx, 3), round(wz, 3), par.fell(r["samples"])))
print(sorted(out, key=lambda t: t[2]))
