"""Reduction uses an absolute error budget scaled by sum(abs(x))."""

TOLERANCES = {
    "float32": {"rtol": 2e-5, "atol": 2e-6},
    "float16": {"rtol": 2e-3, "atol": 2e-4},
    "bfloat16": {"rtol": 2e-2, "atol": 2e-3},
}


def assert_correct(actual, expected, inputs, workload, *, rtol=None, atol=None):
    import torch

    x = inputs[0]
    tolerance = TOLERANCES[str(x.dtype).split(".")[-1]].copy()
    if workload == "reduction":
        # Output is always FP32. Avoid a relative-only test near cancellation.
        # This is an engineering error budget, not a bound for arbitrary N/data.
        scale = x.double().abs().sum().item()
        tolerance = {"rtol": 2e-5, "atol": max(2e-6, 8 * 2**-23 * scale)}
    if rtol is not None:
        tolerance["rtol"] = rtol
    if atol is not None:
        tolerance["atol"] = atol
    torch.testing.assert_close(actual, expected, equal_nan=True, **tolerance)
