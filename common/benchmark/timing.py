"""Two explicit timing scopes; never compare speedups across them."""

import math
import statistics
import time


def percentile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("Need samples and a quantile in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(samples):
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("Latency samples must be finite and positive")
    return {
        "median": statistics.median(samples),
        "p20": percentile(samples, 0.2),
        "p50": percentile(samples, 0.5),
        "p80": percentile(samples, 0.8),
        "mean": statistics.mean(samples),
        "stddev": statistics.pstdev(samples),
    }


def measure(fn, synchronize, *, warmup=25, repetitions=100, mode="wall", device=None):
    if warmup < 1 or repetitions < 2:
        raise ValueError("Require warmup >= 1 and repetitions >= 2")
    if mode not in {"wall", "cuda_event"}:
        raise ValueError(f"Unknown timing mode: {mode}")
    # Compilation, extension loading, lazy initialization, and warmup are excluded.
    result = fn()
    synchronize(result)
    for _ in range(warmup):
        result = fn()
        synchronize(result)
    samples = []
    if mode == "cuda_event":
        import torch

        with torch.cuda.device(device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            # Materialize lazy CUDA events before collecting samples.
            start.record()
            end.record()
            end.synchronize()
            for _ in range(repetitions):
                start.record()
                result = fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
    else:
        for _ in range(repetitions):
            synchronize(result)
            start_ns = time.perf_counter_ns()
            result = fn()
            synchronize(result)
            samples.append((time.perf_counter_ns() - start_ns) / 1000)
    return {"latency_us": summarize(samples), "samples_us": samples}
