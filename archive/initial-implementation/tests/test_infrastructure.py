import json
from pathlib import Path

import pytest

from benchmarks.report import markdown, render_csv, rows
from benchmarks.run_all import load_cases
from common.benchmark.metrics import derived_metrics, work_model
from common.benchmark.timing import measure, percentile, summarize
from common.python.environment import collect
from common.python.registry import WORKLOADS

ROOT = Path(__file__).resolve().parents[1]


def test_percentiles():
    assert percentile([40, 10, 30, 20], 0.2) == pytest.approx(16)
    assert summarize([10, 20, 30, 40])["median"] == 25
    for samples in ([], [0], [-1], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            summarize(samples)


def test_timer_excludes_initialization_and_warmup():
    calls, fences = [], []
    result = measure(lambda: calls.append(1), lambda x: fences.append(x), warmup=3, repetitions=5)
    assert len(calls) == 1 + 3 + 5
    assert len(fences) == 1 + 3 + 2 * 5
    assert len(result["samples_us"]) == 5


def test_metrics_units():
    model = work_model("vector_add", [1000], 4)
    metrics = derived_metrics(model, 2, 100)
    assert metrics["effective_bandwidth_gbs"] == pytest.approx(6)
    assert metrics["percent_peak_bandwidth"] == pytest.approx(6)
    assert metrics["flops_per_second"] == pytest.approx(5e8)
    assert work_model("reduction", [1000], 2)["minimum_bytes"] == 2004
    assert work_model("softmax", [2, 3], 2)["flops"] is None


def test_reporting_separates_timing_scopes():
    # Synthetic test fixture only, never committed as benchmark evidence.
    base = {
        "kernel": "vector_add",
        "shape": [5],
        "dtype": "float32",
        "status": "ok",
        "latency_us": {"median": 10, "p20": 9, "p80": 11},
        "metrics": {"effective_bandwidth_gbs": 1},
    }
    payload = {
        "schema_version": 1,
        "environment": {},
        "results": [
            {**base, "implementation": "pytorch", "timing_mode": "wall"},
            {**base, "implementation": "triton", "timing_mode": "cuda_event"},
        ],
    }
    table = rows(payload)
    assert table[0]["speedup_vs_pytorch"] == 1
    assert table[1]["speedup_vs_pytorch"] is None
    assert "cuda_event" in markdown(payload)
    assert "median_us" in render_csv(payload)
    payload["results"] = []
    assert "Results pending hardware benchmark" in markdown(payload)


def test_environment_is_serializable():
    metadata = collect()
    assert metadata["python"] and metadata["os"]
    json.dumps(metadata, allow_nan=False)


def test_repository_structure():
    for package in WORKLOADS.values():
        directory = ROOT.joinpath(*package.split("."))
        for file in (
            "README.md",
            "benchmark.py",
            "test.py",
            "pytorch_impl.py",
            "jax_impl.py",
            "triton_impl.py",
            "cuda_impl.py",
            "cuda/naive.cu",
            "cuda/optimized.cu",
            "cuda/bindings.cpp",
            "cuda/CMakeLists.txt",
            "results/README.md",
        ):
            assert (directory / file).is_file(), file
    assert len(list((ROOT / "kernels").glob("[0-9][0-9]_*/README.md"))) == 14
    for config in (ROOT / "benchmarks/configs").glob("*.json"):
        assert load_cases(config)
