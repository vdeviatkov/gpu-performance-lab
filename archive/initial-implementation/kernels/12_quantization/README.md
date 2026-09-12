# Quantization and dequantization

**Status: planned — no implementation or performance claim.**

## Problem and scope

Begin with symmetric per-row INT8 quantization with a precisely defined scale, clipping, and rounding rule.

## Candidate baselines and implementations

PyTorch reference; Triton reduction/conversion; CUDA packed stores; JAX only with matching semantics.

## Performance hypothesis

Fuse scale reduction with conversion when beneficial; distinguish bandwidth savings from numerical accuracy loss.

## Correctness acceptance criteria

All-zero groups, saturation, ties, outliers, odd group sizes, NaN/Inf policy, and reconstruction error.

## Promotion criteria

Define tensor/precision contracts; implement and test the PyTorch reference;
add the simplest useful custom backend; integrate shape/dtype configs with the
[shared runner](../../benchmarks/run_all.py); document traffic and resource
reasoning; then add a named optimization with benchmark and profiler evidence.
Follow the [contribution guide](../../CONTRIBUTING.md). Avoid adding stubs that
can be mistaken for executable implementations.

## Results

Results pending hardware benchmark
