import pytest

from common.correctness.checks import assert_correct

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("n", [0, 1, 31, 32, 33, 255, 256, 257, 4099, 1048583])
def test_sum(n, backend, dtype, execute):
    torch.manual_seed(11)
    x = torch.randn(n, device=backend[1], dtype=dtype)
    actual = execute("reduction", x)
    assert actual.dtype == torch.float32 and actual.shape == ()
    assert_correct(actual, x.sum(dtype=torch.float32), (x,), "reduction")
    assert_correct(actual, x.double().sum().float(), (x,), "reduction")


@pytest.mark.parametrize("kind", ["zeros", "ones", "cancellation", "dynamic_range"])
def test_adversarial(kind, backend, dtype, execute):
    values = {
        "zeros": [0.0] * 4099,
        "ones": [1.0] * 4099,
        "cancellation": [128.0, 0.25, -128.0, -0.25] * 1024 + [0.5],
        "dynamic_range": [1024.0, 2**-10, -1024.0, 2**-10] * 1024,
    }
    x = torch.tensor(values[kind], device=backend[1], dtype=dtype)
    actual = execute("reduction", x)
    expected = x.sum(dtype=torch.float32)
    assert_correct(actual, expected, (x,), "reduction")
    if kind in {"zeros", "ones", "cancellation"}:
        torch.testing.assert_close(actual, x.double().sum().float(), rtol=0, atol=0)
