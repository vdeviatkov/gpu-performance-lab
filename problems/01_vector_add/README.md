# 01 · Vector Addition

## Source

LeetGPU: [Vector Addition](https://leetgpu.com/challenges/vector-addition)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Stage

Stage 1 — GPU Fundamentals

Prerequisites: None; start here.

## Priority

**P1** — important.

## Difficulty

- LeetGPU: **Easy**.
- Curriculum: **1/5**, using the independent [difficulty scale](../../docs/curriculum.md#difficulty-scale).

## Why this problem matters

The smallest useful experiment separates launch costs from streaming throughput and establishes a reproducible baseline for the rest of the lab.

## Primary lesson

**Launch overhead vs bandwidth**

## Primary concepts

- Indexing
- Memory coalescing
- Vectorized access
- Launch overhead

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
01_vector_add/
    README.md
    cuda/
    triton/
    pytorch/
    jax/
    tests/
    benchmarks/
    results/
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. One element per thread with explicit tails.
2. Grid-stride work distribution.
3. Aligned vector accesses with a safe scalar tail.
4. Sweep launch geometry and working sets larger than L2.

## Correctness focus

Empty and tiny lab cases, odd lengths, storage offsets, matching dtypes, and aliasing rules.

## Things to measure

- Latency percentiles
- Effective GB/s
- Load/store instruction count
- DRAM and L2 throughput

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Where does launch overhead stop dominating?
- Does vectorization reduce instructions when scalar accesses are already coalesced?

## Status

⬜ **Planned**

An [archived prototype](../../archive/README.md) exists, but is not a completed curriculum implementation.

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update [progress](../../docs/progress.md) manually.
