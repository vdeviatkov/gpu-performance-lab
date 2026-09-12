# Progress

Manually maintained. Last scaffold review: 2026-09-12.

## Overall

**0 / 50 complete** · 0 in progress · 0 optimized · 50 planned.

Completion means acceptance against the current curriculum contract, not the
mere existence of an old prototype. Vector addition, reduction, and softmax
have [archived drafts](../archive/README.md); none has been promoted or validated
as a completed study in this roadmap.

## By stage

| Stage | Complete | In progress | Optimized |
|---|---:|---:|---:|
| 1 — GPU Fundamentals | 0 / 6 | 0 | 0 |
| 2 — Memory & Data Movement | 0 / 6 | 0 | 0 |
| 3 — Parallel Primitives | 0 / 8 | 0 | 0 |
| 4 — Matrix Operations | 0 / 9 | 0 | 0 |
| 5 — ML Primitives | 0 / 8 | 0 | 0 |
| 6 — Transformer Kernels | 0 / 8 | 0 | 0 |
| 7 — Portfolio Extensions | 0 / 5 | 0 | 0 |

## Backend coverage

Count accepted backend implementations only; CUDA baseline and optimized
variants count as one backend per study. Optimized is a subset of Complete.

| Backend | Accepted / planned |
|---|---:|
| CUDA | 0 / 50 |
| Triton | 0 / 48 |
| PyTorch | 0 / 50 |
| JAX | 0 / 29 |

## Source coverage

- LeetGPU: 0 / 43 complete.
- Portfolio Extensions: 0 / 7 complete (two embedded matrix studies, five final capstones).

## Flagship kernels

**0 / 14 complete** · 0 / 14 optimized.

- [ ] [08 · Matrix Transpose](../problems/08_matrix_transpose/README.md)
- [ ] [13 · Reduction](../problems/13_reduction/README.md)
- [ ] [24 · Batched Matrix Multiplication](../problems/24_batched_matmul_fp32/README.md)
- [ ] [25 · General Matrix Multiplication (GEMM)](../problems/25_gemm_fp16/README.md)
- [ ] [32 · Softmax](../problems/32_softmax/README.md)
- [ ] [33 · RMS Normalization](../problems/33_rms_norm/README.md)
- [ ] [34 · Layer Normalization](../problems/34_layer_norm/README.md)
- [ ] [36 · Fused Residual Add and RMS Norm](../problems/36_fused_residual_rms_norm/README.md)
- [ ] [39 · SwiGLU MLP Block](../problems/39_swiglu_mlp/README.md)
- [ ] [42 · Multi-Head Attention](../problems/42_multi_head_attention/README.md)
- [ ] [44 · Grouped Query Attention](../problems/44_grouped_query_attention/README.md)
- [ ] [47 · KV-Cache Update and Paged Access](../problems/47_kv_cache_operations/README.md)
- [ ] [48 · FlashAttention-Style Online Attention](../problems/48_flash_attention/README.md)
- [ ] [49 · Paged Attention](../problems/49_paged_attention/README.md)

## Update procedure

When starting a study, update its README and the root roadmap to 🟨 In Progress.
When all planned backends and correctness requirements are satisfied, mark it
✅ Complete and update this page. If backend scope changes, record the reason
and adjust denominators before claiming completion. A partial backend does not
make the whole problem complete.

Use 🚀 Optimized only after measured optimization evidence, profiling-based
analysis, and reproducibility notes are present. Optimized still counts once
in the complete totals. Keep the flagship checklist consistent. No automation
or dashboard is needed while this remains a small manually curated curriculum.
