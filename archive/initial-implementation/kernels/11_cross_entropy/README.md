# Cross entropy

**Status: planned — no implementation or performance claim.**

## Problem and scope

Compute stable log-sum-exp minus target logits with explicit reduction and ignore-index semantics.

## Candidate baselines and implementations

PyTorch cross_entropy; JAX; Triton row/tiled kernels; CUDA stable reductions.

## Performance hypothesis

Fuse softmax/log/reduction work without materializing probabilities; wide vocabularies require tiled stable reductions.

## Correctness acceptance criteria

Invalid/ignored targets, all ignored, extreme logits, large vocabularies, reduction modes, and optional smoothing semantics.

## Promotion criteria

Define tensor/precision contracts; implement and test the PyTorch reference;
add the simplest useful custom backend; integrate shape/dtype configs with the
[shared runner](../../benchmarks/run_all.py); document traffic and resource
reasoning; then add a named optimization with benchmark and profiler evidence.
Follow the [contribution guide](../../CONTRIBUTING.md). Avoid adding stubs that
can be mistaken for executable implementations.

## Results

Results pending hardware benchmark
