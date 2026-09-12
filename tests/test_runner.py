"""Failures must never become benchmark numbers or dependency skips."""

from types import SimpleNamespace

import pytest

from benchmarks import run_all
from common.python.registry import BackendUnavailable

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("failure", ["incorrect", "compile", "unavailable"])
def test_correctness_gate_and_failure_classification(monkeypatch, failure):
    x = torch.ones(7)
    args = SimpleNamespace(
        device="cpu", timing="wall", warmup=1, repetitions=2, peak_bandwidth=None
    )

    def prepare(*_):
        if failure == "unavailable":
            raise BackendUnavailable("test dependency unavailable")
        if failure == "compile":
            raise RuntimeError("test compiler failure")
        return lambda: torch.zeros_like(x), lambda _: None, lambda y: y

    def must_not_measure(*_, **__):
        pytest.fail("An incorrect/unavailable implementation reached timing")

    monkeypatch.setattr(run_all, "prepare", prepare)
    monkeypatch.setattr(run_all, "measure", must_not_measure)
    record = run_all.run_case({"kernel": "vector_add", "shape": [7]}, "pytorch", args, (x, x))
    assert record["status"] == ("skipped" if failure == "unavailable" else "failed")
    assert "latency_us" not in record and "samples_us" not in record
