# Flagship studies

These 14 P0 workloads receive the deepest treatment. Priority reflects the
quality of the engineering story, not source difficulty: transpose and reduction
are as important here as advanced attention.

| Problem | Stage | Primary engineering story |
|---|---|---|
| [08 · Matrix Transpose](../problems/08_matrix_transpose/README.md) | S2 | Shared-memory bank conflicts |
| [13 · Reduction](../problems/13_reduction/README.md) | S3 | Warp-level reduction |
| [24 · Batched Matrix Multiplication](../problems/24_batched_matmul_fp32/README.md) | S4 | Batch scheduling and occupancy |
| [25 · General Matrix Multiplication (GEMM)](../problems/25_gemm_fp16/README.md) | S4 | Tensor Core utilization |
| [32 · Softmax](../problems/32_softmax/README.md) | S5 | Numerically stable fused reduction |
| [33 · RMS Normalization](../problems/33_rms_norm/README.md) | S5 | Normalization traffic and precision |
| [34 · Layer Normalization](../problems/34_layer_norm/README.md) | S5 | Stable variance reduction |
| [36 · Fused Residual Add and RMS Norm](../problems/36_fused_residual_rms_norm/README.md) | S5 | Residual-normalization fusion |
| [39 · SwiGLU MLP Block](../problems/39_swiglu_mlp/README.md) | S6 | MLP fusion boundaries |
| [42 · Multi-Head Attention](../problems/42_multi_head_attention/README.md) | S6 | Head layout and scheduling |
| [44 · Grouped Query Attention](../problems/44_grouped_query_attention/README.md) | S6 | Shared K/V head reuse |
| [47 · KV-Cache Update and Paged Access](../problems/47_kv_cache_operations/README.md) | S7 | Stateful cache layout |
| [48 · FlashAttention-Style Online Attention](../problems/48_flash_attention/README.md) | S7 | Numerically stable online reduction |
| [49 · Paged Attention](../problems/49_paged_attention/README.md) | S7 | Irregular decode-time attention |

## Required evidence package

- Explicit source/lab contract, error budget, and adversarial correctness cases.
- Straightforward baseline plus several meaningful named optimization variants.
- Matched PyTorch and Triton comparisons; JAX where listed in the roadmap.
- Shape, dtype, layout, and workload-distribution matrices, including slowdowns.
- Raw latency samples with useful percentiles, source commit, and environment.
- Nsight Systems timeline for launches/gaps and focused Nsight Compute captures.
- Traffic/arithmetic model, lower-bound reasoning, and applicable roofline analysis.
- An explanation connecting counter changes to the hypothesized bottleneck.
- Resource tradeoffs: registers, occupancy, shared memory, synchronization, and spills.
- Reproducible commands, limitations, negative results, and a next falsifying experiment.

Name variants for their mechanism: `shared_tree`, `warp_shuffle`, `padded_tile`,
`register_tile`, or `online_softmax`. A name describes an experiment, not evidence
of a speedup. Avoid adding sophisticated mechanisms with no measured need.

## Suggested review route

A short portfolio review should start at **08 Transpose → 13 Reduction →
25 FP16 GEMM → 32 Softmax → 39 SwiGLU MLP → 48 Online Attention**.
A deeper inference review adds **44 GQA → 47 KV cache → 49 Paged Attention**.
The complete ordering and prerequisites remain in the main roadmap.

## Status

0 / 14 complete. No flagship is currently supported by NVIDIA measurement
or profiling evidence in the active curriculum. Earlier prototypes remain
separate in the [archive](../archive/README.md).
