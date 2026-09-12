import pytest

from common.correctness.checks import assert_correct

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("n", [0, 1, 3, 4, 31, 255, 256, 257, 1025, 1048583])
def test_add(n, backend, dtype, execute):
    torch.manual_seed(7)
    a = torch.randn(n, device=backend[1], dtype=dtype)
    b = torch.randn_like(a)
    result = execute("vector_add", a, b)
    assert_correct(result, a + b, (a, b), "vector_add")


@pytest.mark.parametrize("offset", [1, 2, 3, 4])
def test_contiguous_storage_offset(offset, backend, dtype, execute):
    a = torch.arange(1037, device=backend[1], dtype=torch.float32).to(dtype)[offset:1029]
    b = torch.ones_like(a)
    assert a.is_contiguous()
    assert_correct(execute("vector_add", a, b), a + b, (a, b), "vector_add")
