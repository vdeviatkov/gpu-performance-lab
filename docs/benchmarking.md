# Benchmarking plan

**Plan only. No active benchmark framework is implemented in this scaffold.**

## Contract before timing

Every comparison should specify shape, strides/layout, dtype and accumulator,
output precision, mutation, approximations, and what is included in the call.
Use the same input values and a meaningful PyTorch/library baseline. Add
`torch.compile` as a separately named baseline where fusion is relevant. JAX
comparisons must preserve semantics and use a GPU rather than an accidental CPU
fallback.

Correctness gates timing; follow the [test requirements](methodology.md#correctness-and-tests)
before collecting performance samples.

## Planned measurement rules

- Complete an initialization call, compilation, extension build, and warmup
  before collecting steady-state samples. Report cold-start costs separately.
- Use GPU completion correctly. CUDA events must bracket work on the relevant
  stream. Host timing needs a completion fence, not just an asynchronous launch.
- With JAX, JIT-compile outside timing and wait for results with
  `block_until_ready()`. A PyTorch event pair must not pretend to time another
  runtime's untracked stream. See [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html).
- Distinguish device/stream elapsed time from host-observed API latency. Include
  dispatch, allocations, transfers, or synchronization only under an explicit
  scope; never mix scopes into one speedup table.
- Report microsecond latency where appropriate, median/p50, p20/p80, sample
  count, and optionally mean/standard deviation. Keep raw samples.
- Warm up each backend, repeat complete experiments, vary backend order, and
  record the random seed. Percentiles are not confidence intervals.
- Sweep multiple shapes and supported FP32/FP16/BF16 cases. Add integer formats
  only when the source or lab extension defines rounding and scaling.
- Include both small launch-sensitive and large working sets. Separate reused
  inputs from a streaming/rotating-buffer policy, and compare footprint to L2.
- Benchmark multi-pass algorithms and fused pipelines end to end, including all
  required passes. Restore mutable inputs/cache state consistently between samples.

PyTorch exposes CUDA events and explicit completion facilities; their scope
must match the measured work. Consult the [CUDA semantics guide](https://docs.pytorch.org/docs/stable/notes/cuda.html)
when implementing the timer. Profiling and ordinary timing should be separate runs.

## Metrics and their limits

| Metric | Planned use | Interpretation limit |
|---|---|---|
| Latency and percentiles | All workloads | State device-only versus API/end-to-end scope |
| Effective GB/s | Copy, transpose, reduction, fusion, caches | Useful/logical bytes divided by time are not measured DRAM throughput |
| FLOP/s | GEMV/GEMM and explicitly modeled attention | State useful versus executed work, precision, masks, and padding |
| Elements/nonzeros/tokens per second | Primitives, sparse kernels, routing | Define exactly what one item represents |
| Arithmetic intensity | Roofline reasoning | Specify the traffic model and memory level |
| Percentage of theoretical bandwidth | Sourced hardware ceiling | Do not guess the ceiling or equate cache throughput with DRAM bandwidth |
| Occupancy/register/shared-memory use | Explain resource constraints | These are profiler observations, not independent performance scores |
| Numerical error and memory footprint | Mixed precision and fusion | A faster result with changed semantics is a different experiment |

## Hardware and setup

Choose and record a concrete GPU/toolchain when implementation begins. No device
support matrix or executable setup is established yet. Plan for Python 3.11+,
modern C++17 or later, compatible PyTorch/Triton/JAX packages, and a matching
NVIDIA driver/toolkit. A single documented GPU is enough to start; CPU reference
checks are not GPU performance evidence.

Record compute capability, SM count, memory capacity and L2 size, and sourced
bandwidth/compute ceilings for the exact arithmetic mode. FP16/BF16, integer
matrix instructions, FP8, and asynchronous copies depend on the actual hardware
and software. Record input, accumulator, and output dtypes separately. Tensor
Core support does not prove efficient utilization.

Add a small extension/build setup once there is code to compile. Save an exact
package freeze with each measured run; avoid a large dependency stack or Docker
until it solves a concrete reproducibility problem.

## Environment record

Future JSON should record GPU model/UUID and architecture, driver, toolkit and
runtime versions, PyTorch/Triton/JAX versions, Python/OS, exact package freeze,
source commit/dirty state, relevant environment flags, device/stream, clocks,
power/thermal/load notes, dtype/layout, seed, timing/warmup/sample settings,
allocation/cache policies, correctness status, and raw latency samples.

Use JSON as the source of truth; later create Markdown, CSV, and plots from it.
Record unavailable capabilities as explicit skips and distinguish them from
compiler, correctness, out-of-memory, or runtime failures. Never assign a timing
to a failed or skipped implementation.

## Honest comparisons

Pair speedups only across matching workload semantics, hardware, input cases,
precision, and timing scope. Publish noise and slowdowns. For stateful inference,
separate GPU kernel latency from host scheduling and memory-management costs,
then provide an end-to-end view where it answers the systems question.

Do not commit synthetic timings or illustrative Nsight counters as results.
Before a real run, use exactly: **Results pending hardware benchmark**.
