# KV cache operations

**Status: planned — no implementation or performance claim.**

## Problem and scope

Implement append and gather for a contiguous cache before adding block/page indirection.

## Candidate baselines and implementations

PyTorch indexing reference; Triton; CUDA; JAX only if mutation/copy semantics can be compared fairly.

## Performance hypothesis

Study contiguous writes, indexed reads, layout choice, cache reuse, and per-token launch overhead.

## Correctness acceptance criteria

Duplicate slots, bounds, sequence lengths, page boundaries, ownership rules, and aliasing.

## Promotion criteria

Define tensor/precision contracts; implement and test the PyTorch reference;
add the simplest useful custom backend; integrate shape/dtype configs with the
[shared runner](../../benchmarks/run_all.py); document traffic and resource
reasoning; then add a named optimization with benchmark and profiler evidence.
Follow the [contribution guide](../../CONTRIBUTING.md). Avoid adding stubs that
can be mistaken for executable implementations.

## Results

Results pending hardware benchmark
