# Layer normalization

**Status: planned — no implementation or performance claim.**

## Problem and scope

Normalize each row using its mean and variance, then apply affine scale and bias.

## Candidate baselines and implementations

PyTorch layer_norm; JAX where useful; Triton row reduction; CUDA cooperative rows.

## Performance hypothesis

Fuse statistics and affine work to reduce global traffic; compare variance algorithms and register retention.

## Correctness acceptance criteria

Constant rows, small variance, odd widths, epsilon semantics, FP32 accumulation, and affine broadcasting.

## Promotion criteria

Define tensor/precision contracts; implement and test the PyTorch reference;
add the simplest useful custom backend; integrate shape/dtype configs with the
[shared runner](../../benchmarks/run_all.py); document traffic and resource
reasoning; then add a named optimization with benchmark and profiler evidence.
Follow the [contribution guide](../../CONTRIBUTING.md). Avoid adding stubs that
can be mistaken for executable implementations.

## Results

Results pending hardware benchmark
