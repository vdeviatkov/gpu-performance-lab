# Curriculum guide

The [roadmap](../README.md#roadmap) is the single overview of the 50 studies and
their progress. Problem READMEs hold source links, prerequisites, planned
backends, and experiments. The [skill map](skills.md) offers an alternative
way to browse, including a short list of flagship studies.

## Difficulty scale

| Level | Intended engineering depth |
|---|---|
| 1/5 | Basic indexing, elementwise work, ownership, launch boundaries |
| 2/5 | Memory mapping, instruction behavior, simple aggregation |
| 3/5 | Shared memory, warp communication, moderate tiling |
| 4/5 | Multi-pass algorithms, stable ML reductions, substantial tiling and fusion |
| 5/5 | Tensor Core tuning, online attention, stateful inference, sophisticated fusion |

This describes the intended study depth, not just the first correct kernel.
Official LeetGPU difficulty is separate and appears in each problem README.

## Learning route

Follow indexing and ownership with copy/transpose and tiled memory access.
Implement reduction before count, losses, dot product, scan, and compaction.
Then progress through GEMV, sparse access, FP32 GEMM, batching, FP16/Tensor Cores,
quantized math, normalization, fusion, attention, caches, and routing.

Start with **01 Vector Addition → 02 ReLU → 03 Reverse Array → 07 Matrix Copy →
08 Matrix Transpose**, then **13 Reduction**. Other introductory studies are
useful focused exercises. Prerequisite links let you choose a route without
maintaining a separate stage or priority system.

CUDA exposes hardware mapping; Triton provides a compiler/block-layout contrast;
PyTorch supplies the reference and a meaningful baseline. Add JAX where XLA
lowering or fusion answers a distinct question. The roadmap records backend
scope; no separate coverage table needs updating.

## Sources and scope

The [LeetGPU catalog](https://leetgpu.com/challenges) was inspected on 2026-09-12:
43 challenges were selected from 99 listings, with titles, URLs, and official
difficulties verified. Seven Portfolio Extensions fill gaps or extend workloads
into systems studies. Each problem README is the source-link record; there is
no duplicate catalog table.

Keep these source distinctions explicit when implementing:

- **Parallel Merge** is the catalog match for merge-sorted-arrays.
- **Batched Matrix Multiplication** uses FP32; **General Matrix Multiplication
  (GEMM)** uses FP16 matrices with FP32 alpha/beta scalars.
- RMSNorm, LayerNorm, RoPE, and fused residual + RMSNorm already have source
  challenges. Source RMSNorm uses a vector with scalar scale/shift; the fused
  residual challenge uses weighted rows.
- BatchNorm reduces over the batch axis. Cross entropy consumes logits and class
  indices. SwiGLU activation splits a vector into halves; the MLP is a separate
  projection workload.
- Weight Dequantization receives scale metadata. The custom quantization study
  also computes scales and converts values.
- Decaying Causal Attention is unnormalized; introducing softmax changes its contract.

Confirm current source signatures before implementation. Label broader dtype,
layout, precision, or batching experiments as lab variants. Link statements
rather than copying them or solutions.

## Why this selection

The core limits duplicate activations, dimensional counting variants, and general
2D/3D convolution. Copy provides a control for transpose; 1D convolution teaches
halos; Gaussian blur studies conditional separability. FP32 batching, FP16 GEMM,
and FP16 batching isolate scheduling, matrix instructions, and tile granularity.

Full transformer blocks, additional attention masks, FFT/graph algorithms, and
application-level regression/clustering are deferred to keep individual hardware
hypotheses visible. Useful next electives include segmented scans/SSMs, top-k or
top-p sampling, INT4 GEMM, quantized KV attention, and attention backward.

The earlier implementation remains available in Git history at commit
`dd2d199`; it is not part of this planning scaffold or accepted performance evidence.
