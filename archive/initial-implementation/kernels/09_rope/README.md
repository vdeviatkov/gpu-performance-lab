# Rotary position embeddings

**Status: planned — no implementation or performance claim.**

## Problem and scope

Rotate feature pairs using position-dependent sine/cosine tables and an explicit pairing/layout convention.

## Candidate baselines and implementations

PyTorch reference; Triton; CUDA vector loads; JAX if it provides a meaningful XLA comparison.

## Performance hypothesis

Coalescing and table reuse versus launch overhead; consider fusion with projection/cache writes only after a standalone baseline.

## Correctness acceptance criteria

Interleaved versus split-half pairing, positions, partial rotary dimensions, odd/invalid widths, and dtype rounding.

## Promotion criteria

Define tensor/precision contracts; implement and test the PyTorch reference;
add the simplest useful custom backend; integrate shape/dtype configs with the
[shared runner](../../benchmarks/run_all.py); document traffic and resource
reasoning; then add a named optimization with benchmark and profiler evidence.
Follow the [contribution guide](../../CONTRIBUTING.md). Avoid adding stubs that
can be mistaken for executable implementations.

## Results

Results pending hardware benchmark
