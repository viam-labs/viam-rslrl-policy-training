# /// script
# dependencies = ["viam-sdk>=0.85"]
# ///
"""Smoke test of the walker on the microduck-preview machine: stand 10 s, walk at vx 0.4 for 5 s, stop.

    uv run scripts/duck_smoke.py <cell.json> <out_dir>
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, "/Users/nick.hehr/.claude/plugins/cache/viam-mujoco-sim/mujoco-sim/0.1.6/skills/mujoco-sim-cell/scripts")
import cell  # noqa: E402
from viam.components.camera import Camera  # noqa: E402
from viam.components.generic import Generic  # noqa: E402
from viam.services.generic import Generic as GenericService  # noqa: E402


async def main(cell_json, out):
    out.mkdir(parents=True, exist_ok=True)
    async with cell.connect(cell_json) as robot:
        world, duck = Generic.from_robot(robot, "sim-world"), Generic.from_robot(robot, "duck")
        walker = GenericService.from_robot(robot, "walker")
        cam = Camera.from_robot(robot, "overview")
        # order matters: the policy must take over within a few ticks of play
        await world.do_command({"command": "pause"})
        await world.do_command({"command": "reset"})
        await walker.do_command({"command": "set_command", "vx": 0, "vy": 0, "wz": 0})
        await world.do_command({"command": "play"})
        print("start:", await walker.do_command({"command": "start"}))
        await asyncio.sleep(10)
        st = await walker.do_command({"command": "status"})
        print("standing status:", {k: st.get(k) for k in ("state", "ticks", "tick_ms_p50", "tick_ms_p99", "late", "warmup_ms", "reason")})
        s = await duck.do_command({"io": {}})
        print("gravity z:", s["gravity"][2])
        assert st["state"] == "running" and 450 <= st["ticks"] <= 510 and st["tick_ms_p99"] < 20 and s["gravity"][2] < -0.95, "standing check"
        x0 = (await duck.do_command({"io": {}})).get("trunk_pos", [None])[0]
        await walker.do_command({"command": "set_command", "vx": 0.4})
        await asyncio.sleep(2.5)
        images, _ = await cam.get_images()
        (out / "walk-mid.jpg").write_bytes(images[0].data)
        print("image:", out / "walk-mid.jpg")
        await asyncio.sleep(2.5)
        s = await duck.do_command({"io": {}})
        print(f"trunk x before {x0} after {s.get('trunk_pos', [None])[0]}; gravity z {s['gravity'][2]}")
        assert s["trunk_pos"][0] - x0 > 0.3, "walking check: moved < 0.3 m"
        print("walking status:", await walker.do_command({"command": "status"}))
        print("stop:", await walker.do_command({"command": "stop"}))
        await world.do_command({"command": "pause"})


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
