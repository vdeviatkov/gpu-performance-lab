"""Algorithmic traffic lower bounds, not measured DRAM traffic."""

import math


def work_model(workload, shape, itemsize):
    n = math.prod(shape)
    if workload == "vector_add":
        return {"elements": n, "minimum_bytes": 3 * n * itemsize, "flops": n}
    if workload == "reduction":
        return {"elements": n, "minimum_bytes": n * itemsize + 4, "flops": max(0, n - 1)}
    if workload == "softmax":
        # Exponentials, max comparisons, and divisions do not have a useful single
        # FLOP cost; avoid a misleading FLOP/s number for this workload.
        return {"elements": n, "minimum_bytes": 2 * n * itemsize, "flops": None}
    raise ValueError(workload)


def derived_metrics(model, latency_us, peak_bandwidth_gbs=None):
    if latency_us <= 0 or (peak_bandwidth_gbs is not None and peak_bandwidth_gbs <= 0):
        raise ValueError("Latency and optional peak bandwidth must be positive")
    seconds = latency_us * 1e-6
    bandwidth = model["minimum_bytes"] / seconds / 1e9
    return {
        "effective_bandwidth_gbs": bandwidth,
        "elements_per_second": model["elements"] / seconds,
        "flops_per_second": None if model["flops"] is None else model["flops"] / seconds,
        "percent_peak_bandwidth": (
            None if peak_bandwidth_gbs is None else 100 * bandwidth / peak_bandwidth_gbs
        ),
    }
