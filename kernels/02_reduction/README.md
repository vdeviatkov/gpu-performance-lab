# Reduction

## Problem

Compute the sum of a contiguous 1-D FP32/FP16/BF16 tensor. Every backend
accumulates and returns a scalar in FP32. Empty input returns FP32 zero.
Floating-point association may differ, so bitwise agreement with PyTorch is not
required for arbitrary values. No autograd or strided-input support is hidden
behind copies.

## Baseline

`torch.sum(x, dtype=torch.float32)` is the production/reference operation. The
custom CUDA baseline is a correct parallel reduction with coalesced input reads,
so the optimized comparison isolates communication within blocks.

## Implementations

| Backend | Strategy |
|---|---|
| PyTorch | Native sum with FP32 accumulation/output |
| JAX/XLA | JIT-compiled FP32 sum |
| Triton | Contiguous tiles, `tl.sum`, hierarchical FP32 partial buffers |
| CUDA naive: `shared_tree` | Grid-stride accumulation, shared tree, two kernel launches |
| CUDA optimized: `warp_shuffle` | Same grid geometry; warp shuffles and shared warp totals, two launches |

## Algorithm

CUDA launches 256-thread blocks, capped at 4,096 blocks. Each thread accumulates
a grid-stride subsequence in FP32. Stage one writes one FP32 partial per block;
stage two reduces those partials with one block. There are no global atomics,
host reads, or CPU synchronization between stages. Both launches run on the
current PyTorch stream and scratch lifetime is managed by PyTorch.

The baseline writes 256 values to shared memory and uses a sequential-addressing
tree with barriers. The optimized version uses `__shfl_down_sync` inside each
warp, writes eight warp totals to shared memory, synchronizes the block once,
and uses the first warp to reduce those eight values. Every lane in each
participating warp executes the shuffle; inactive data lanes contribute zero.
The aggregate is consumed only by thread zero.

Triton uses a default tile of 1,024 elements and four warps. Masks insert the
additive identity for tail elements. It recursively reduces partial arrays until
one value remains, so very large inputs may take more than two launches. Tiles
256/512/1024/2048 and 4/8 warps are explicit options; no autotuning changes the
configuration during measurements.

## Performance model

The minimum traffic is `Ns + 4` bytes and useful work approximately `N-1` adds,
giving low arithmetic intensity near `1/s` FLOP/byte. The CUDA two-pass algorithm
actually moves approximately `Ns + 8P + 4` bytes, where `P` is the partial count.
Triton adds read/write traffic for each partial level. The runner's effective
bandwidth uses the lower bound; all passes are included in measured latency.

Large inputs are bandwidth candidates. Small inputs can be dominated by the
extra launch and allocation. Accumulation dependency chains, synchronization,
and the small second-stage grid can limit throughput independently of bandwidth.
There is no universal claim that this is always memory bound.

## Optimization journey

### Optimization 1 — warp-local communication

**What changed:** replace the shared-memory tree inside each block with register
shuffles, one shared value per warp, and one cross-warp barrier.

**Why it should help:** fewer shared operations and block-wide barriers reduce
communication overhead. Both versions already read coalesced data and avoid
interleaved-address bank conflicts. The expected gain is primarily reduced
synchronization, not a claim of fixing bank conflicts that were never present.

**Benchmark result:** Results pending hardware benchmark.

**Profiling evidence:** pending. Compare barrier stalls, shared transactions,
eligible warps per cycle, register count, and throughput at equal launch geometry.
Check both passes. Higher occupancy alone is insufficient evidence of a gain.

## Benchmark methodology

```bash
python -m kernels.02_reduction.benchmark --config benchmarks/configs/full.json \
  --implementations pytorch triton cuda_naive cuda_optimized \
  --timing cuda_event --output artifacts/reduction.json
python -m pytest kernels/02_reduction -q
bash scripts/profile.sh ncu --kernel reduction --implementation cuda_optimized --shape 16777219
```

See [shared timing rules](../../docs/benchmarking.md). Tests cover warp/block/tile
boundaries, empty input, large odd lengths, constants, cancellation, and dynamic
range. A FP64 diagnostic reference complements the FP32 PyTorch comparison.
The configurable absolute error budget scales with `sum(abs(x))`; it is an
engineering acceptance rule rather than a proof for arbitrary input distributions.

## Results

Results pending hardware benchmark

## Analysis

No measured winner is established. A strong result would show when fewer
barriers matter and when memory traffic or the second launch dominates. Compare
Triton's pass count before attributing a difference only to intra-block code.

## Limitations

FP32 summation is not compensated or associative. The fixed CUDA block cap is a
portable starting point rather than an occupancy-tuned choice. Both CUDA variants
retain two launches even for tiny nonempty vectors; scratch is allocated per
call. GPU correctness, synchronization sanitizers, and profiles remain pending.

## Next experiments

Add a one-block small-input path, compare block count against SM count, then
study multiple independent accumulators per thread to shorten dependency chains.
Benchmark a CUB baseline before introducing more elaborate custom algorithms.

## Reference

[NVIDIA warp-level primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
explains participation masks and register exchange. The lab's code is a small
independent implementation with explicit full-warp participation.
