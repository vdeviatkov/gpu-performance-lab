# SwiGLU

**Status: planned — no implementation or performance claim.**

## Problem and scope

Compute SiLU(gate) times up, with a documented packed or separate input layout.

## Candidate baselines and implementations

PyTorch eager/compiled; JAX; Triton fused pointwise; CUDA scalar/vector paths.

## Performance hypothesis

Fuse activation and multiplication to remove temporary traffic and a launch; inspect special-function throughput.

## Correctness acceptance criteria

Large positive/negative inputs, zeros, tails, packed alignment, dtype error, and output aliasing restrictions.

## Promotion criteria

Define tensor/precision contracts; implement and test the PyTorch reference;
add the simplest useful custom backend; integrate shape/dtype configs with the
[shared runner](../../benchmarks/run_all.py); document traffic and resource
reasoning; then add a named optimization with benchmark and profiler evidence.
Follow the [contribution guide](../../CONTRIBUTING.md). Avoid adding stubs that
can be mistaken for executable implementations.

## Results

Results pending hardware benchmark
