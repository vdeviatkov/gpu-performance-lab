# Experimental methodology

## A falsifiable optimization claim

Every optimization should answer five questions:

1. What bottleneck do we expect for a particular shape, dtype, and GPU?
2. What changes in the work mapping, memory traffic, or instructions?
3. Why should that change improve execution on this hardware?
4. Does repeated benchmark evidence show the predicted improvement?
5. Do profiler counters support the proposed explanation?

Until measurements exist, describe expected behavior as a hypothesis. Preserve
baselines and negative results. Report slowdowns and crossover sizes; do not
publish only a selected winning shape. Keep algorithmic changes distinct from
precision, approximation, and semantics changes.

## Roofline reasoning

Arithmetic intensity is useful work divided by traffic at a specified memory
level. A simple bound is `min(peak_compute, bandwidth * arithmetic_intensity)`.
Specify whether bandwidth means DRAM, L2, or shared memory. Count rereads,
temporary buffers, and intermediate reduction passes when estimating a concrete
implementation. A lower-bound traffic model is useful for comparing algorithms
but does not replace measured traffic.

Vector add has intensity `1/(3s)` FLOP/byte, so a large streaming input is a
bandwidth candidate. Tiny vectors expose launch overhead. Reductions add
dependencies, barriers, and extra launches to a low-intensity operation. Softmax
also requires exponentials and reductions, so instruction/dependency throughput
can matter even when its minimum traffic is small.

## Resource reasoning

Coalescing concerns addresses issued by neighboring lanes. Vector loads may
reduce instructions without improving already coalesced transactions. Shared
memory can reduce cross-warp communication cost, but barriers and bank conflicts
can offset that gain. Registers retain values cheaply until pressure reduces
resident warps or induces local-memory spills. Occupancy alone does not predict
performance; examine eligible warps, issue activity, and memory latency together.

Tensor Cores, asynchronous copies, cooperative groups, and persistent scheduling
belong in experiments whose data reuse and synchronization requirements justify
them. The first three workloads do not need these mechanisms to make their
optimization questions clear. Future GEMM/attention work will introduce them
with architecture and precision contracts.

## Numerical reasoning

Reduction order changes floating-point rounding. Tests compare against both
PyTorch FP32 and FP64 diagnostics, with exact checks for selected representable
patterns. The generic sum error budget scales with `sum(abs(x))` so cancellation
does not make relative error meaningless; this budget is not a mathematical
guarantee for unbounded data or tensor size. Also examine absolute error against
FP64 for any performance-focused reduction change.

Softmax subtracts the row maximum before exponentiation. Rows containing NaN,
positive infinity, or only negative infinity follow PyTorch's NaN behavior;
tests explicitly check those cases. No fast-math compiler flag is enabled.
Triton's exponential approximation is checked with dtype-specific tolerances.

## Evidence checklist for a measured result

- Clean source commit, exact package freeze, GPU/driver/toolkit inventory.
- Full command, raw samples, seed, output policy, clocks/power, thermal/load notes.
- Correctness suite and relevant race/memory checks passing on that GPU.
- Shape/dtype sweep with matching PyTorch comparison and dispersion.
- Focused profiler capture with interpretation and plausible alternative causes.
- Limits and a next experiment that could disprove the interpretation.
