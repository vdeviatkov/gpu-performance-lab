# Benchmarking

## Contract and fairness

The runner compares functional, forward-only APIs: input creation, host/device
transfer, DLPack conversion, JIT compilation, and extension builds are outside
timing. Output allocation and reduction scratch allocation occur inside each
call for every backend. Eager PyTorch is the production baseline. A future
`torch.compile` comparison must get a separate name and compilation warmup.

All implementations receive the same seeded input values, shape, and dtype.
Vector add/reduction use one-dimensional contiguous tensors; softmax uses
contiguous matrices and the last axis. Reduction accumulates and returns FP32.
Softmax uses stable max subtraction and FP32 intermediate arithmetic for the
custom kernels. Dtype-specific tolerances and a cancellation-aware reduction
budget live in `common/correctness/checks.py`; callers can override tolerances.

## Timing scopes

| Mode | Completion mechanism | Meaning | Backends |
|---|---|---|---|
| `wall` (default) | `torch.cuda.synchronize(device)` or JAX `block_until_ready()` | Host-observed functional API latency, including dispatch/allocation/fence overhead | All |
| `cuda_event` | Events recorded on the current PyTorch CUDA stream; end event synchronized | Elapsed stream interval around one functional call | PyTorch, Triton, CUDA |

CUDA events avoid charging the final host synchronization to the interval, but
Python submission gaps may still appear between start/end events for very short
kernels. This is not a pure kernel-duration measurement and does not amortize
launch costs with CUDA Graphs. Use Nsight Systems to separate dispatch gaps from
GPU execution. JAX runs on its own runtime/stream, so a PyTorch event pair must
not time it. Do not compare speedups between modes or CPU/GPU runs.

The initial call is completed before 25 warmup iterations; the default then
collects 100 individually completed samples. Each sample is one invocation.
Timing includes all reduction passes. Samples are retained in microseconds,
with interpolated p20/p50/p80, median, population standard deviation, and mean.
Percentiles describe observed variation, not confidence intervals.

## Running a study

1. Run correctness tests and confirm backend versions/toolkit compatibility.
2. Record environment and idle GPU conditions. Set clocks/power only if authorized
   on the machine; record the chosen settings and thermal state.
3. Use `smoke.json` to check execution, then `full.json` for launch-sensitive,
   cache-resident, and larger working sets. Check available VRAM first.
4. Repeat complete runs with different recorded seeds and separate output paths.
   The seed controls input generation and shuffled backend order within each case.
5. Compare distributions and repeat anomalous cases. Control external GPU load.
6. Profile representative cases separately from timing. Profiler replay changes
   execution and cache state; profiler duration is not the benchmark latency.

Repeated inputs intentionally provide a warm/reused working set. There is no
cache flush or rotating input pool. A large vector case does not prove streaming
DRAM behavior on every GPU: compare its footprint with that GPU's L2 size and
inspect counters. Rotating buffers and CUDA Graph timing are future methodology
extensions, each requiring a distinct recorded timing/cache policy.

## Derived metrics

For `N` values with `s` bytes/value, vector add has minimum traffic `3Ns` and `N`
adds. Reduction has `Ns + 4` minimum bytes and approximately `N-1` additions,
excluding intermediate traffic and identity additions. Softmax has a lower bound
of `2Ns` bytes. Its exponentials, comparisons, and divides are not summarized by
an arbitrary FLOP count. Each kernel README explains actual pass traffic.

Effective bandwidth is `minimum_bytes / seconds / 1e9` in decimal GB/s.
It is an algorithmic throughput metric, not a measured memory-bus rate. Values
above nominal DRAM bandwidth can reflect cache residency. Supply
`--peak-bandwidth VALUE --bandwidth-source URL` to record a sourced hardware
ceiling and compute its percentage; no hardware ceiling is guessed.

## Result format

Schema version 1 has `environment`, `settings`, `cache_policy`, `allocation_policy`,
and `results`. Each successful record has kernel/shape/dtype/implementation,
variant, timing mode, correctness status, raw `samples_us`, `latency_us`, work
model, and derived metrics. Skips/failures have a reason and no latency numbers.
Compiler, OOM, launch, and correctness errors fail the run; unavailable optional
dependencies/capabilities are explicit skips. `--require-all` rejects skips too.

The runner writes completed records atomically after each implementation and
refuses to overwrite an existing run. Reports operate on one JSON run to prevent
accidental mixing of hardware/environments. A PyTorch speedup is emitted only
for matching shape, dtype, workload, and timing mode within that run.

```bash
python -m benchmarks.run_all --kernel softmax --shape 4096 4096 \
  --dtypes fp16 bf16 --warmup 50 --repetitions 200 --output artifacts/softmax-run1.json
python -m benchmarks.report artifacts/softmax-run1.json --output artifacts/softmax-run1.md
python -m benchmarks.report artifacts/softmax-run1.json --format csv --output artifacts/softmax-run1.csv
python -m pip freeze > artifacts/requirements-measured.txt
```

Use CSV in a plotting tool, grouping by shape/dtype/mode and showing percentile
bands. Never mix different devices or timing scopes into one speedup curve.

## References

- [PyTorch CUDA semantics and timing](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [JAX asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)
- [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html)
