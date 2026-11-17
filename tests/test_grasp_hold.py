"""The held peg must stay put in the grasp on MuJoCo Warp (CPU here; the probe runs it on GPU).

settle: the first 0.5 s after reset (no knock when the fingers close); creep: the next 2 s.
"""

from rl_insert.task.checks import grasp_hold_slip, make_env


def test_grasp_holds_for_2s():
    env = make_env(num_envs=64, device="cpu")
    slip = grasp_hold_slip(env, seconds=2.0, settle_s=0.5)
    print(f"max slip along TCP z (mm): {slip}")
    assert slip["settle_mm"] < 0.5
    assert slip["creep_mm"] < 0.5
