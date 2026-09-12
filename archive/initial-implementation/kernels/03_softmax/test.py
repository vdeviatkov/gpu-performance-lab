import pytest

from common.correctness.checks import assert_correct

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "shape", [(0, 7), (1, 1), (3, 7), (5, 31), (17, 129), (8, 1025), (2, 8193), (32, 32768)]
)
def test_softmax(shape, backend, dtype, execute):
    torch.manual_seed(13)
    x = torch.randn(shape, device=backend[1], dtype=dtype)
    actual = execute("softmax", x)
    assert_correct(actual, torch.softmax(x, -1), (x,), "softmax")
    torch.testing.assert_close(
        actual.float().sum(-1), torch.ones(shape[0], device=x.device), rtol=0, atol=0.01
    )


@pytest.mark.parametrize(
    "kind", ["constant", "large", "negative_inf", "all_negative_inf", "positive_inf", "nan"]
)
def test_pathological(kind, backend, dtype, execute):
    values = {
        "constant": [5.0] * 7,
        "large": [10000.0, 0.0, -10000.0, 9990.0, -9990.0, 0.1, -0.1],
        "negative_inf": [0.0, -float("inf"), -2.0, 2.0, 1.0, 0.0, -1.0],
        "all_negative_inf": [-float("inf")] * 7,
        "positive_inf": [float("inf"), 1.0, 0.0, -2.0, 1.0, 0.0, -1.0],
        "nan": [float("nan"), 1.0, 0.0, -2.0, 1.0, 0.0, -1.0],
    }
    x = torch.tensor([values[kind]], device=backend[1], dtype=dtype)
    assert_correct(execute("softmax", x), torch.softmax(x, -1), (x,), "softmax")
