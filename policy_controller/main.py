import asyncio

from viam.module.module import Module

from policy_controller.service import PolicyController  # noqa: F401  (registers the model)

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
