# Vector addition

## Problem

Compute `c[i] = a[i] + b[i]` for two contiguous 1-D tensors with identical device,
shape, and dtype. Output preserves FP32/FP16/BF16 dtype. Empty vectors and
contiguous offset views are supported; broadcasting, strided inputs, and autograd
are outside the contract.

## Baseline

Eager `torch.add` is both the reference and production baseline. This workload
tests the benchmark's ability to separate launch-sensitive sizes from streaming
memory behavior. A custom implementation is not assumed to beat PyTorch.

## Implementations

| Backend | Mapping | Precision |
|---|---|---|
| PyTorch | `a + b` | Input/output dtype |
| JAX/XLA | `jax.jit(a + b)` on prepared device arrays | Input/output dtype |
| Triton | 1024 contiguous elements/program, 4 warps, masked tail | FP32 arithmetic; store to input dtype |
| CUDA naive: `scalar_coalesced` | 256 threads/block, grid-stride scalar loop | FP32 arithmetic; store to input dtype |
| CUDA optimized: `float4_or_scalar_fallback` | Aligned FP32 groups of four, scalar tail | FP32; scalar fallback for FP16/BF16 or unaligned input |

CUDA grid size is capped at 65,535 blocks; 64-bit grid-stride indexing covers
larger inputs. `float4` requires 16-byte aligned input pointers. PyTorch outputs
have allocator alignment; storage-offset inputs are checked explicitly. The
lower-precision fallback is intentional and must not be described as vectorized.

## Algorithm

The scalar kernel maps adjacent lanes to adjacent elements. The vector path maps
each lane to four consecutive FP32 values and uses `float4` loads/stores, with
up to three tail elements handled separately. No shared memory, barriers, or
inter-thread communication is necessary. Shared staging would add work without
reuse. Triton uses program IDs to select contiguous tiles and masks all boundary
loads and stores; block sizes 256/512/1024/2048 and 4/8 warps are exposed for
controlled experiments, without automatic tuning in the benchmark.

## Performance model

For `N` elements and `s` bytes/value, useful work is `N` additions and minimum
traffic is `3Ns` bytes: two reads and one write. Arithmetic intensity is
`1/(3s)` FLOP/byte: `1/12` for FP32 and `1/6` for FP16/BF16. A sufficiently large
streaming vector should primarily be memory-bandwidth bound; small vectors are
likely dominated by launch/dispatch latency. Reused cache-resident data targets
a different memory level.

The runner reports effective bandwidth as `3Ns / time`. This is not measured
DRAM bandwidth. Vectorization changes instruction count, not the traffic model.

## Optimization journey

### Optimization 1 — aligned vector memory operations

**What changed:** each CUDA thread moves four FP32 elements per vector iteration.
The scalar baseline already has coalesced accesses, so coalescing is not the
hypothesized improvement.

**Why it should help:** fewer load/store/address instructions per element may
reduce instruction issue pressure. Larger register demand and less per-element
parallelism can counteract that benefit. A DRAM-saturated scalar kernel may see
little gain.

**Benchmark result:** Results pending hardware benchmark.

**Profiling evidence:** pending. Compare SASS load/store widths, instruction
count, register count, occupancy, DRAM/L2 traffic, and achieved throughput. Verify
the compiler retained vector operations. Use offset tests to confirm fallback
correctness and measure the fallback separately when studying alignment.

## Benchmark methodology

```bash
python -m kernels.01_vector_add.benchmark --config benchmarks/configs/full.json \
  --dtypes fp32 fp16 bf16 --output artifacts/vector.json
python -m pytest kernels/01_vector_add -q
bash scripts/profile.sh ncu --kernel vector_add --implementation cuda_optimized --shape 67108864
```

The shared [methodology](../../docs/benchmarking.md) excludes initialization and
reports synchronized steady-state samples. Benchmarks use aligned tensors;
correctness tests additionally cover offsets 1–4, vector tails, zero, one,
non-power-of-two, and million-element inputs.

## Results

Results pending hardware benchmark

## Analysis

No speedup claim is supported yet. A useful result will locate the transition
from launch-sensitive to cache-sensitive to DRAM-sensitive sizes and determine
whether vectorization changes instruction count at that transition. FP16/BF16
CUDA variant comparisons are a fallback control, not independent optimizations.

## Limitations

No backward pass, broadcasting, strided layouts, half2/BF16 packed arithmetic,
CUDA Graph timing, or cache-flushing mode. CPU references validate semantics and
tooling only; CUDA/Triton require hardware validation.

## Next experiments

Measure instruction width and crossover sizes; add a controlled aligned/offset
benchmark sweep; then consider packed FP16/BF16 operations if counters identify
instruction pressure. Add rotating input buffers for a distinct streaming study.
