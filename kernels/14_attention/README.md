# Attention

**Status: planned — no implementation or performance claim.**

## Problem and scope

Start with stable scaled dot-product attention with explicit causal mask and layout contracts.

## Candidate baselines and implementations

PyTorch scaled_dot_product_attention with backend selection recorded; JAX; Triton; CUDA after GEMM/softmax validation.

## Performance hypothesis

Establish a materialized baseline; later use tiled online softmax to reduce quadratic intermediate traffic.

## Correctness acceptance criteria

Causal boundaries, non-square lengths, head dimensions, all-masked rows, dtype error, and matched mask semantics.

## Promotion criteria

Define tensor/precision contracts; implement and test the PyTorch reference;
add the simplest useful custom backend; integrate shape/dtype configs with the
[shared runner](../../benchmarks/run_all.py); document traffic and resource
reasoning; then add a named optimization with benchmark and profiler evidence.
Follow the [contribution guide](../../CONTRIBUTING.md). Avoid adding stubs that
can be mistaken for executable implementations.

## Results

Results pending hardware benchmark
