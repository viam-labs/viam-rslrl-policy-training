import asyncio

from viam.module.module import Module

from duck_policy.service import DuckPolicy  # noqa: F401  (registers the model)
from duck_policy.teleop import Teleop  # noqa: F401  (registers the model)

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
