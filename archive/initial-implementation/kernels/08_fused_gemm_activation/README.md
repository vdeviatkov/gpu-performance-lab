# Fused GEMM and activation

**Status: planned — no implementation or performance claim.**

## Problem and scope

Compute an explicitly specified GEMM epilogue such as bias plus ReLU or SiLU.

## Candidate baselines and implementations

PyTorch eager/compiled; JAX/XLA; Triton epilogue; CUDA once the GEMM baseline is validated.

## Performance hypothesis

Avoid intermediate output traffic and launches; measure when epilogue fusion matters relative to matrix compute.

## Correctness acceptance criteria

Bias broadcasting, odd matrix dimensions, activation extremes, output dtype, and matched approximation semantics.

## Promotion criteria

Define tensor/precision contracts; implement and test the PyTorch reference;
add the simplest useful custom backend; integrate shape/dtype configs with the
[shared runner](../../benchmarks/run_all.py); document traffic and resource
reasoning; then add a named optimization with benchmark and profiler evidence.
Follow the [contribution guide](../../CONTRIBUTING.md). Avoid adding stubs that
can be mistaken for executable implementations.

## Results

Results pending hardware benchmark
