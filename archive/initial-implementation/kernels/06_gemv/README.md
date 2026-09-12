# Matrix-vector multiplication

**Status: planned — no implementation or performance claim.**

## Problem and scope

Compute y = A x, initially for explicit contiguous row-major layouts.

## Candidate baselines and implementations

PyTorch matmul; JAX/XLA comparison; Triton tiled reductions; CUDA warp/block mappings.

## Performance hypothesis

Bandwidth and matrix layout dominate at low reuse; compare per-row and split reductions across aspect ratios.

## Correctness acceptance criteria

Tall/wide matrices, odd leading dimensions, mixed accumulation precision, zero dimensions, and alignment.

## Promotion criteria

Define tensor/precision contracts; implement and test the PyTorch reference;
add the simplest useful custom backend; integrate shape/dtype configs with the
[shared runner](../../benchmarks/run_all.py); document traffic and resource
reasoning; then add a named optimization with benchmark and profiler evidence.
Follow the [contribution guide](../../CONTRIBUTING.md). Avoid adding stubs that
can be mistaken for executable implementations.

## Results

Results pending hardware benchmark
