# 12 · Weight Dequantization

## Source

LeetGPU: [Weight Dequantization](https://leetgpu.com/challenges/weight-dequantization)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Stage

Stage 2 — Memory & Data Movement

Prerequisites: [07 · Matrix Copy](../07_matrix_copy/README.md); [08 · Matrix Transpose](../08_matrix_transpose/README.md)

## Priority

**P1** — important.

## Difficulty

- LeetGPU: **Medium**.
- Curriculum: **3/5**, using the independent [difficulty scale](../../docs/curriculum.md#difficulty-scale).

## Why this problem matters

The source operation applies a scale grid to a weight matrix. It teaches reuse of small metadata without assuming an INT4 packing contract that the page does not specify.

## Primary lesson

**Scale-tile locality**

## Primary concepts

- Data layouts
- Cache behavior
- Scale metadata

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Planned locations, created only when real work starts:

```text
12_weight_dequantization/
    README.md
    cuda/
    triton/
    pytorch/
    tests/
    benchmarks/
    results/
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Direct element-to-scale lookup.
2. Tile work to reuse scale values.
3. Compare scale broadcast and repeated cached loads.
4. Study edge tiles and vectorized output.

## Correctness focus

Tile-size divisibility, source input/output types, scale indexing, and partial edge tiles.

## Things to measure

- Latency
- Useful GB/s
- Scale traffic and L2 behavior
- Registers/thread

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Do repeated scale loads already hit cache?
- Does scale reuse justify changing an otherwise coalesced mapping?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update [progress](../../docs/progress.md) manually.
