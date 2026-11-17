"""PegInsertRunner's fail-fast and best-checkpoint logic, without a training run."""
import pytest
import torch

from rl_insert.task.runner import PegInsertRunner, success_rate


def _runner(fail_fast_iters=300):
    r = object.__new__(PegInsertRunner)  # skip __init__: no env needed for _watch
    r.ema, r.ever_succeeded, r.best, r.fail_fast_iters = None, False, -1.0, fail_fast_iters
    return r


def test_success_rate_counts_ends_across_steps():
    steps = [
        {"Episode_Termination/success": torch.tensor(1.0), "Episode_Termination/excess_force": torch.tensor(3.0)},
        {"Episode_Termination/success": torch.tensor(0.0), "Episode_Termination/time_out": torch.tensor(0.0)},
        {"Curriculum/difficulty": torch.tensor(0.3)},
    ]
    assert success_rate(steps) == pytest.approx(0.25)
    assert success_rate([{"Curriculum/difficulty": 0.3}]) is None


def test_fail_fast_only_when_nothing_ever_succeeded():
    r = _runner(fail_fast_iters=10)
    for it in range(10):
        r._watch(it, 0.0, 0.3)
    with pytest.raises(RuntimeError, match="fail fast"):
        r._watch(10, 0.0, 0.3)
    r = _runner(fail_fast_iters=10)
    r._watch(3, 0.01, 0.3)
    for it in range(4, 50):
        r._watch(it, 0.0, 0.3)  # one early success is enough
    r = _runner(fail_fast_iters=0)
    r._watch(10_000, None, 0.3)  # disabled


def test_ema_only_counts_full_difficulty():
    r = _runner()
    r._watch(0, 0.95, 0.9)
    assert r.ema is None  # easier levels don't count
    r._watch(1, 0.6, 1.0)
    assert r.ema == 0.6
    r._watch(2, 0.9, 0.95)
    r._watch(3, 0.7, 1.0)
    assert r.ema == 0.7  # restarts on re-entry to full difficulty


def test_best_copied_on_improvement_with_a_fresh_onnx(tmp_path, monkeypatch):
    r = _runner()
    run = tmp_path / "run"
    run.mkdir()
    export_ok = [True]

    def fake_save(self, path, infos=None):
        open(path, "w").write(path)
        if export_ok[0]:
            (run / "run.onnx").write_text(path)  # what the export would hold

    monkeypatch.setattr(PegInsertRunner.__mro__[1], "save", fake_save)
    for it, ema, ok in ((100, 0.5, True), (150, 0.4, True), (200, 0.9, False)):
        r.ema, r.current_learning_iteration, export_ok[0] = ema, it, ok
        r.save(str(run / f"model_{it}.pt"))
    # 150 is worse; 200 is better but its export failed, so 100 stays best, as a matched pair
    assert (run / "best.pt").read_text().endswith("model_100.pt")
    assert (run / "best.onnx").read_text().endswith("model_100.pt")
    assert '"iteration": 100' in (run / "best.json").read_text()
    assert r.best == 0.5
