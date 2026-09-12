# Matrix multiplication

**Status: planned — no implementation or performance claim.**

## Problem and scope

Compute C = A B with an explicit layout, dtype, accumulation, and epilogue contract.

## Candidate baselines and implementations

PyTorch/cuBLAS baseline; JAX/XLA; Triton tiled GEMM; incremental CUDA SIMT and Tensor Core paths.

## Performance hypothesis

Move from scalar to shared tiling to register tiles; then introduce Tensor Cores and asynchronous copies when justified.

## Correctness acceptance criteria

Rectangular and odd M/N/K, tail tiles, strides, accumulation tolerances, and architecture capability checks.

## Promotion criteria

Define tensor/precision contracts; implement and test the PyTorch reference;
add the simplest useful custom backend; integrate shape/dtype configs with the
[shared runner](../../benchmarks/run_all.py); document traffic and resource
reasoning; then add a named optimization with benchmark and profiler evidence.
Follow the [contribution guide](../../CONTRIBUTING.md). Avoid adding stubs that
can be mistaken for executable implementations.

## Results

Results pending hardware benchmark
