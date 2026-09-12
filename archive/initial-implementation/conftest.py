"""Fixtures shared by numbered workload tests and infrastructure tests."""

import pytest

from benchmarks.run_all import prepare
from common.python.registry import BackendUnavailable

BACKENDS = [
    pytest.param(("pytorch", "cpu"), id="pytorch-cpu"),
    pytest.param(("jax", "cpu"), id="jax-cpu", marks=pytest.mark.jax),
    *[
        pytest.param((name, "cuda:0"), id=f"{name}-cuda", marks=pytest.mark.gpu)
        for name in ("pytorch", "jax", "triton", "cuda_naive", "cuda_optimized")
    ],
]


@pytest.fixture(params=BACKENDS)
def backend(request):
    import torch

    name, device = request.param
    if device.startswith("cuda") and not torch.cuda.is_available():
        if request.config.getoption("--require-gpu-backends", default=False):
            pytest.fail("CUDA GPU required by --require-gpu-backends")
        pytest.skip("NVIDIA CUDA GPU unavailable")
    return name, device


@pytest.fixture(params=["float32", "float16", "bfloat16"])
def dtype(request, backend):
    import torch

    if request.param == "bfloat16" and backend[1].startswith("cuda"):
        if not torch.cuda.is_bf16_supported():
            pytest.skip("BF16 unsupported by hardware")
    return getattr(torch, request.param)


@pytest.fixture
def execute(backend, request):
    def run(workload, *inputs):
        try:
            fn, synchronize, convert = prepare(workload, backend[0], inputs)
            result = fn()
            synchronize(result)
            return convert(result)
        except BackendUnavailable as exc:
            if backend[1].startswith("cuda") and request.config.getoption(
                "--require-gpu-backends", default=False
            ):
                pytest.fail(str(exc))
            pytest.skip(str(exc))

    return run


def pytest_addoption(parser):
    parser.addoption("--require-gpu-backends", action="store_true", default=False)
