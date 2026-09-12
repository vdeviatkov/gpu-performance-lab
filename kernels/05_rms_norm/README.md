# RMS normalization

**Status: planned — no implementation or performance claim.**

## Problem and scope

Scale a row by the reciprocal square root of its mean square plus epsilon, then apply weights.

## Candidate baselines and implementations

PyTorch expression/reference; JAX; Triton; CUDA warp/block reduction.

## Performance hypothesis

A simpler fused reduction than LayerNorm; examine one-read versus reread implementations and register pressure.

## Correctness acceptance criteria

Zero rows/values, wide dynamic range, epsilon placement, odd widths, FP16/BF16 error, and weight dtype.

## Promotion criteria

Define tensor/precision contracts; implement and test the PyTorch reference;
add the simplest useful custom backend; integrate shape/dtype configs with the
[shared runner](../../benchmarks/run_all.py); document traffic and resource
reasoning; then add a named optimization with benchmark and profiler evidence.
Follow the [contribution guide](../../CONTRIBUTING.md). Avoid adding stubs that
can be mistaken for executable implementations.

## Results

Results pending hardware benchmark
