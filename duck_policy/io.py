"""How duck-policy reaches the duck. SimIO talks to viam-labs:mujoco-sim's robot io DoCommand; a BusIO
(later, on the real duck) will talk to the Dynamixel bus. Units: rad, rad/s, trunk frame; gravity is a
unit vector, upright [0, 0, -1]."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Snapshot:
    t: float
    q: list[float]
    dq: list[float]
    gyro: list[float]
    gravity: list[float]
    generation: int
    torque: bool
    trunk_pos: list[float] | None = None


class DuckIO(Protocol):
    async def read(self) -> Snapshot: ...

    async def write(self, targets: list[float]) -> None: ...

    async def torque(self, on: bool) -> Snapshot: ...


class SimIO:
    """The duck robot's io DoCommand. Checks the joint order once, against the policy's."""

    def __init__(self, duck: Any, joints: list[str]) -> None:
        self._duck, self._joints, self._checked = duck, joints, False

    async def _io(self, request: dict) -> Snapshot:
        r = await self._duck.do_command({"io": request})
        if not self._checked:
            if list(r["joints"]) != self._joints:
                raise ValueError(f"duck joints {list(r['joints'])} != the policy's {self._joints}")
            self._checked = True
        return Snapshot(t=float(r["t"]), q=list(r["q"]), dq=list(r["dq"]), gyro=list(r["gyro"]),
                        gravity=list(r["gravity"]), generation=int(r["generation"]), torque=bool(r["torque"]),
                        trunk_pos=list(r.get("trunk_pos", [])) or None)

    async def read(self) -> Snapshot:
        return await self._io({})

    async def write(self, targets: list[float]) -> None:
        await self._io({"targets": [float(v) for v in targets]})

    async def torque(self, on: bool) -> Snapshot:
        return await self._io({"torque": bool(on)})
