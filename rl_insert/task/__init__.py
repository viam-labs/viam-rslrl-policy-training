"""PegInsert-xArm6 mjlab task. Importing this registers it."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import TaskParams, peg_insert_env_cfg
from .runner import PegInsertRunner, peg_insert_ppo_cfg

TASK_ID = "PegInsert-xArm6"

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=peg_insert_env_cfg(),
    play_env_cfg=peg_insert_env_cfg(play=True),
    rl_cfg=peg_insert_ppo_cfg(),
    runner_cls=PegInsertRunner,
)

__all__ = ["TASK_ID", "TaskParams", "peg_insert_env_cfg", "PegInsertRunner", "peg_insert_ppo_cfg"]
