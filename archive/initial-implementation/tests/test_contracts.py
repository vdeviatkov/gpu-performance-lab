import pytest

from common.python.contracts import validate
from common.python.registry import BackendUnavailable, implementation

torch = pytest.importorskip("torch")


def test_rejects_invalid_inputs():
    invalid = [
        torch.ones(2, 3),
        torch.ones(8)[::2],
        torch.ones(4, dtype=torch.int32),
        torch.ones(4, requires_grad=True),
    ]
    for x in invalid:
        with pytest.raises((ValueError, TypeError)):
            validate(x, ndim=1)
    with pytest.raises(ValueError):
        validate(torch.ones(2, 0), ndim=2)
    with pytest.raises(ValueError):
        validate(torch.ones(2), ndim=1, other=torch.ones(3))
    with pytest.raises(ValueError):
        validate(torch.ones(2), ndim=1, cuda=True)


@pytest.mark.gpu
@pytest.mark.parametrize("workload", ["vector_add", "reduction", "softmax"])
@pytest.mark.parametrize("backend", ["cuda_naive", "cuda_optimized", "triton"])
def test_current_stream(workload, backend, request):
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA GPU unavailable")
    try:
        fn = implementation(workload, backend)
    except BackendUnavailable as exc:
        if request.config.getoption("--require-gpu-backends"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        shape = (17, 129) if workload == "softmax" else (4099,)
        x = torch.randn(shape, device="cuda")
        args = (x, torch.randn_like(x)) if workload == "vector_add" else (x,)
        actual = fn(*args)
        expected = implementation(workload, "pytorch")(*args)
        # Consume on the same nondefault stream to catch incorrect launch streams.
        error = (actual - expected).abs().max()
    stream.synchronize()
    assert error.item() < 1e-3


@pytest.mark.gpu
@pytest.mark.parametrize("workload", ["vector_add", "reduction", "softmax"])
@pytest.mark.parametrize("variant", ["cuda_naive", "cuda_optimized"])
def test_cuda_contract_rejection(workload, variant):
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA GPU unavailable")
    fn = implementation(workload, variant)
    # CUDA wrappers validate their inputs independently of the Python adapter.
    shape = (4, 8) if workload == "softmax" else (16,)
    x = torch.ones(shape, device="cuda", requires_grad=True)
    args = (x, x) if workload == "vector_add" else (x,)
    with pytest.raises(RuntimeError, match="autograd"):
        fn(*args)
    x = x.detach()[..., ::2]
    args = (x, x) if workload == "vector_add" else (x,)
    with pytest.raises(RuntimeError, match="contiguous"):
        fn(*args)


@pytest.mark.gpu
@pytest.mark.parametrize("workload", ["vector_add", "reduction", "softmax"])
def test_cuda_device_guard(workload):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two NVIDIA GPUs required for device guard validation")
    shape = (3, 127) if workload == "softmax" else (1027,)
    with torch.cuda.device(0):
        x = torch.randn(shape, device="cuda:1")
        args = (x, x) if workload == "vector_add" else (x,)
        out = implementation(workload, "cuda_optimized")(*args)
        assert torch.cuda.current_device() == 0
    torch.cuda.synchronize(1)
    torch.testing.assert_close(out, implementation(workload, "pytorch")(*args))
