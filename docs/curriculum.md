# Curriculum design

The ordered route contains **50 core studies: 43 LeetGPU challenges and 7
Portfolio Extensions**. This is a learning and evidence plan, not a promise that
every workload will beat a vendor library. Official source difficulty and our
engineering difficulty are intentionally separate.

## Difficulty scale

| Level | Expected engineering complexity | Representative study |
|---|---|---|
| 1/5 | Basic indexing, elementwise work, ownership, launch boundaries | Vector addition; reverse array |
| 2/5 | Basic memory mapping, instruction behavior, simple aggregation | Interleave; count |
| 3/5 | Shared memory, warp communication, moderate tiling and precision choices | Transpose; reduction; GEMV |
| 4/5 | Multi-stage algorithms, substantial tiling, stable ML reductions, fusion | Compaction; batched FP32 GEMM; softmax |
| 5/5 | Tensor Core/resource tuning, advanced attention, stateful inference, sophisticated fusion | FP16 GEMM; GQA; online attention |

The rating reflects the intended optimized study, not merely writing a correct
first kernel. Difficulty can decrease when a later study isolates a simpler
new idea, such as SiLU or RoPE, after its prerequisites are established.

## Priority scale

- **P0 — flagship:** deep implementation progression and performance explanation.
- **P1 — important:** a distinctive concept or a prerequisite worth implementing carefully.
- **P2 — useful learning:** a bounded experiment that builds a specific skill.

There are 14 P0, 29 P1, and 7 P2 studies. None is marked complete merely because
an archived prototype exists. Completion requires the current study contract
and evidence; see [progress](progress.md).

## Dependency path

Stage 1 introduces indexing, predication, and in-place ownership. Stage 2 moves
through a copy control, transpose, shared tiles, and metadata locality. Stage 3
starts with reduction before count, MSE, dot product, scan, histogram, compaction,
and merge. Count and MSE are deliberately moved out of the introductory stage
because their useful GPU implementations depend on aggregation.

Stage 4 proceeds from dense GEMV and sparse GEMV to FP32 tiled multiplication,
FP32 batching, FP16 GEMM, FP16 batching, quantized matrix math, sparse reuse, and
a fused epilogue. Stage 5 adds stable normalization and fusion. Stage 6 builds
transformer layouts, gated MLPs, materialized attention, causal masks, multiple
heads, positional biases, GQA, and a distinct recurrence-friendly operation.

Stage 7 closes with quantization, cache state, online attention, paging, and
expert dispatch. Dense GEMV and the fused GEMM epilogue are Portfolio Extensions
embedded earlier where their prerequisites and lessons fit. Each problem README
links to earlier prerequisites; no problem depends on a later core entry.

## Implementation depth by backend

| Backend | Planned studies | Role |
|---|---:|---|
| CUDA | 50 | Explicit hardware mapping, straightforward baseline, named optimizations |
| Triton | 48 | Block/program layout, masks, reductions, compiler/resource tradeoffs |
| PyTorch | 50 | Correctness reference and an appropriate production/library baseline |
| JAX | 29 | Selected XLA lowering, fusion, mixed-precision, and attention comparisons |

CUDA-only algorithm depth is sufficient for Rainbow Table and Parallel Merge;
their initial plans defer Triton. Dynamic output, mutable caches, and routing
also defer JAX until a comparable functional contract is worth the effort.
No backend is required just to fill a table cell. Record the reason and update
coverage denominators if scope changes.

The LeetGPU submission interface and the local research API need not be identical.
First reproduce the source contract. Then label wider dtype, shape, stride,
batching, or hardware experiments as lab variants. For example, source FP32
semantics must not quietly become TF32 or lower precision in a comparison.

## Critical selection review

| Review question | Decision |
|---|---|
| Too much trivial elementwise work? | Six introductory studies have different lessons: throughput, predication, ownership, packing, channel layout, integer dependencies. Activation variants beyond SiLU and gating are deferred. |
| Too many convolutions? | Keep 1D convolution for halos and Gaussian blur for conditional separability; defer general 2D/3D convolution. Pooling isolates stride-dependent maximum windows. |
| Reduction and synchronization represented? | Dedicated reduction flagship plus predicate count, map-reduce losses, dot product, scan, histogram, and compaction. |
| Strong memory progression? | Copy control → transpose/bank conflicts → halo reuse → scale metadata → sparse access → paging. |
| Strong matrix progression? | GEMV → sparse GEMV → FP32 reuse → batching → FP16 Tensor Cores → quantized arithmetic → sparse reuse → epilogue fusion. |
| Modern ML relevance? | Norms, residual fusion, RoPE, SwiGLU MLP, GQA, stateful caches, online/paged attention, and MoE dispatch. |
| Distinct advanced lessons? | ALiBi removes bias materialization; GQA studies K/V sharing; decaying attention changes the algorithm; paging introduces indirection and mixed-length scheduling. |
| Triton and XLA represented appropriately? | Triton spans 48 studies; 29 targeted JAX comparisons emphasize compiler/fusion questions rather than mandatory four-backend coverage. |
| Hardware reasoning beyond ML? | Integer dependencies, in-place ownership, irregular merge, sparse load balance, atomics, and cache behavior also serve systems/low-latency learning. |

The next optional expansion should target a new mechanism rather than another
activation or mask: segmented scan/SSM recurrence, INT4 weight-only GEMM,
attention backward, or top-p sampling. See [catalog decisions](catalog.md).
