# Softmax

## Problem

For each row, compute `exp(x - max(x)) / sum(exp(x - max(x)))`. Inputs are
contiguous `[rows, columns]` FP32/FP16/BF16 tensors. Output preserves input dtype;
custom implementations compute max, exponentials, and sum in FP32. Zero rows are
supported; zero columns are rejected. These are forward-only kernels.

## Baseline

`torch.softmax(x, dim=-1)` is the reference and production baseline. The CUDA
baseline retains stable numerics while intentionally using a serial row mapping
to expose the cost of poor lane-to-data mapping and sequential reductions.

## Implementations

| Backend | Strategy |
|---|---|
| PyTorch | Native last-axis softmax |
| JAX/XLA | JIT FP32 softmax followed by cast to input dtype |
| Triton | One padded row/program, max/sum reduction, fused load/exp/store |
| CUDA naive: `serial_row` | One thread/row, three input traversals |
| CUDA optimized: `block_row` | One cooperative 256-thread block/row, warp/block max and sum |

## Algorithm

Both CUDA versions first find the maximum, then accumulate shifted exponentials,
then normalize. They recompute exponentials on the output traversal and reread
inputs instead of allocating a global intermediate or an unbounded register
array. The cooperative version assigns adjacent columns to adjacent lanes;
register shuffles reduce within each warp, shared memory combines eight warp
totals, and barriers publish max/sum before use. A final barrier protects shared
state if a block processes another row.

Triton rounds columns up to a power of two, masks padded loads to negative
infinity, performs `tl.max` and `tl.sum`, and masks stores. Rows up to 32,768
columns are supported; larger widths are explicitly unavailable. Four warps are
used through padded width 2,048 and eight above it; 4/8/16 can be selected
explicitly. These are starting configurations, not measured optima. Large padded
rows can incur significant register pressure or spills.

## Performance model

For `N = rows * columns` and `s` bytes/value, ideal fused traffic is `2Ns` bytes.
The CUDA kernels perform three reads and one write, approximately `4Ns` logical
bytes before cache effects, and evaluate exponentials twice. Triton's source
expresses one read and one write with intermediates retained within the program;
actual traffic must be checked for spills and cache behavior.

The shared report uses the `2Ns` lower bound for effective GB/s, which expresses
useful workload throughput. It does not imply either CUDA kernel attains that
traffic. Softmax has comparisons, exponentials, additions, and divisions, so the
runner does not assign an arbitrary aggregate FLOP count.

Tiny rows can be launch/parallelism limited; large row counts offer more blocks;
wide rows may expose reduction dependencies, exponential throughput, register
pressure, or memory limits. An IO-only roofline is an incomplete model.

## Optimization journey

### Optimization 1 — cooperative row processing

**What changed:** map a row to a CUDA block, use contiguous lane accesses, and
replace serial max/sum with warp/block reductions.

**Why it should help:** neighboring lanes issue coalesced accesses and the row's
reduction work is distributed. Additional barriers and idle lanes on narrow rows
may offset those gains. The baseline's row-per-thread mapping produces strided
addresses across a warp when reading the same column.

**Benchmark result:** Results pending hardware benchmark.

**Profiling evidence:** pending. Compare sectors/requests, achieved occupancy,
eligible warps, barrier stalls, SM throughput, and the row-count/width crossover.

### Optimization 2 — retain row intermediates in Triton

**What changed:** fuse max, exponentiation, sum, and normalization within one
program rather than rereading inputs and recomputing exponentials.

**Why it should help:** lower logical traffic and fewer exponentials if values
stay in registers. Power-of-two padding, register footprint, and spilling may
reverse the benefit at larger widths.

**Benchmark result:** Results pending hardware benchmark.

**Profiling evidence:** pending. Inspect registers/thread, local-memory traffic,
load counts, and instruction throughput. Cross-language comparisons alone do
not isolate fusion; a register-retaining CUDA variant would strengthen causality.

## Benchmark methodology

```bash
python -m kernels.03_softmax.benchmark --config benchmarks/configs/full.json \
  --output artifacts/softmax.json
python -m pytest kernels/03_softmax -q
bash scripts/profile.sh ncu --kernel softmax --implementation triton --shape 4096 4096
```

Follow [shared timing rules](../../docs/benchmarking.md). Tests cover zero rows,
single columns, narrow/odd/wide rows, constants, extreme finite values, mixed
negative infinities, all-negative-infinity rows, positive infinity, and NaN.
Exceptional rows preserve PyTorch's NaN semantics; finite rows are also checked
for normalized sums within an output-dtype allowance.

## Results

Results pending hardware benchmark

## Analysis

No timing or profiler claim has been established. The useful comparisons are
the CUDA mapping change, the cost of rereading versus retaining a row, and
the padding/resource cliff at widths just above a power of two.

## Limitations

No backward pass, strided matrices, masks, attention scaling, or online/tiled
softmax for unbounded rows. The CUDA optimized path is straightforward but still
rereads inputs and recomputes exponentials; it is not intended as a final
production-library replacement. GPU execution remains unvalidated locally.

## Next experiments

Implement a warp-per-row narrow-row specialization, a bounded register-retaining
CUDA path, and a tiled online softmax for wide rows. Measure resource usage before
adding autotuning or persistent scheduling.

## Reference

The [Triton fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
is useful background for program-local fusion and padding; this lab independently
implements a minimal version without persistent scheduling.
