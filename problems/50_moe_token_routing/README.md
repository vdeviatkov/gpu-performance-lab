# 50 · MoE Token Routing and Dispatch

## Source

**Portfolio Extension** — independently scoped laboratory workload, not a LeetGPU challenge.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[17 · Prefix Sum](../17_prefix_sum/README.md); [18 · Histogramming](../18_histogramming/README.md); [19 · Stream Compaction](../19_stream_compaction/README.md); [20 · Parallel Merge](../20_parallel_merge/README.md); [39 · SwiGLU MLP Block](../39_swiglu_mlp/README.md)

## Difficulty

- LeetGPU: **Not applicable — Portfolio Extension**.
- Curriculum: **5/5**, using the independent [difficulty scale](../../docs/curriculum.md#difficulty-scale).

## Why this problem matters

This extension starts with precomputed expert assignments and builds counting, offsets, permutation, and inverse mapping. It goes beyond the catalog Top-K gating task.

## Primary lesson

**Balanced scatter and expert dispatch**

## Primary concepts

- Prefix operations
- Atomics
- Irregular memory access
- Work partitioning
- Kernel fusion

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton are useful for explicit memory ownership and scheduling. JAX is deferred: dynamic sizes or mutation would need a carefully matched functional contract before a fair XLA comparison.

Planned locations, created only when real work starts:

```text
50_moe_token_routing/
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

1. Define dispatch and inverse-permutation contracts.
2. Count expert assignments and form prefix offsets.
3. Scatter tokens with an explicit deterministic-order policy.
4. Study skew, capacity, and fusion of metadata passes.

## Correctness focus

Duplicate assignments, empty experts, capacity overflow policy, inverse mapping, and optional gating as a separate precursor.

## Things to measure

- Routing latency
- Effective bytes and allocations
- Expert load distribution
- Atomic contention and determinism

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- What happens when most tokens select one expert?
- Does a faster scatter violate the required ordering?
- How much time remains outside expert GEMMs?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
