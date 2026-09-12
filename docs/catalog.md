# Catalog provenance and selection

## Verification record

The [live LeetGPU challenge catalog](https://leetgpu.com/challenges) was inspected
on **2026-09-12**. It displayed **99 challenge cards**. All 43 selected challenge
URLs and official difficulty labels below were matched to those rendered cards;
no challenge slug was invented from a proposed problem name. Individual detail
pages were also inspected where naming, precision, or numerical semantics could
change the curriculum design.

The catalog links its [official challenge repository](https://github.com/AlphaGPU/leetgpu-challenges),
but the live cards were the selection inventory. No solution files were copied.
The plans are original engineering summaries; the source links remain the
canonical statements and submission contracts. Dates describe this inspection,
not a guarantee that the catalog will remain unchanged.

## Naming and semantic decisions

- **Vector Add** maps to the official **Vector Addition** title.
- **Merge Sorted Arrays** maps to **Parallel Merge**, a verified live listing.
- **Batched Matrix Multiplication FP32** maps to **Batched Matrix Multiplication**;
  its detail page explicitly specifies FP32.
- **GEMM FP16** maps to **General Matrix Multiplication (GEMM)**. Its detail page
  specifies FP16 A/B/initial-C/output and FP32 alpha/beta scalars. It is not a
  separate invented challenge and is not another FP32 GEMM exercise.
- **Multi-Head Self Attention** maps to **Multi-Head Attention**.
- RMSNorm, LayerNorm, RoPE, and fused residual + RMSNorm already have live source
  challenges. They are recorded as LeetGPU entries, not relabeled as custom work.
- **RMS Normalization** uses a 1D input and scalar scale/shift in the source;
  transformer-style row/feature-weight variants must be labeled separately.
- **Batch Normalization** reduces over the batch axis for each feature; it is
  not interchangeable with row normalization or running-statistics inference.
- **Categorical Cross Entropy Loss** consumes logits and class indices and returns
  a batch-average loss. A probabilities-based loss would be a different contract.
- **Swish-Gated Linear Unit** splits an even-length input vector into two halves;
  **SwiGLU MLP Block** is a separate multi-projection workload.
- **Weight Dequantization** applies a supplied scale grid. It must not silently
  become an INT4 bit-unpacking task or a scale-estimation kernel.
- **Decaying Causal Attention** is unnormalized, geometrically weighted causal
  attention. Its recurrence experiment must not insert a softmax.

## Selected source inventory

Curriculum priority/order is independent of official difficulty. Each local
README links back to the source and supplies its own difficulty and experiments.

| Local study | Verified LeetGPU title and URL | Official difficulty |
|---|---|---|
| [01 · Vector Addition](../problems/01_vector_add/README.md) | [Vector Addition](https://leetgpu.com/challenges/vector-addition) | Easy |
| [02 · ReLU](../problems/02_relu/README.md) | [ReLU](https://leetgpu.com/challenges/relu) | Easy |
| [03 · Reverse Array](../problems/03_reverse_array/README.md) | [Reverse Array](https://leetgpu.com/challenges/reverse-array) | Easy |
| [04 · Interleave Arrays](../problems/04_interleave_arrays/README.md) | [Interleave Arrays](https://leetgpu.com/challenges/interleave-arrays) | Easy |
| [05 · RGB to Grayscale](../problems/05_rgb_to_grayscale/README.md) | [RGB to Grayscale](https://leetgpu.com/challenges/rgb-to-grayscale) | Easy |
| [06 · Rainbow Table](../problems/06_rainbow_table/README.md) | [Rainbow Table](https://leetgpu.com/challenges/rainbow-table) | Easy |
| [07 · Matrix Copy](../problems/07_matrix_copy/README.md) | [Matrix Copy](https://leetgpu.com/challenges/matrix-copy) | Easy |
| [08 · Matrix Transpose](../problems/08_matrix_transpose/README.md) | [Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose) | Easy |
| [09 · 1D Convolution](../problems/09_convolution_1d/README.md) | [1D Convolution](https://leetgpu.com/challenges/1d-convolution) | Easy |
| [10 · Gaussian Blur](../problems/10_gaussian_blur/README.md) | [Gaussian Blur](https://leetgpu.com/challenges/gaussian-blur) | Medium |
| [11 · 2D Max Pooling](../problems/11_max_pooling_2d/README.md) | [2D Max Pooling](https://leetgpu.com/challenges/2d-max-pooling) | Medium |
| [12 · Weight Dequantization](../problems/12_weight_dequantization/README.md) | [Weight Dequantization](https://leetgpu.com/challenges/weight-dequantization) | Medium |
| [13 · Reduction](../problems/13_reduction/README.md) | [Reduction](https://leetgpu.com/challenges/reduction) | Medium |
| [14 · Count Array Element](../problems/14_count_array_element/README.md) | [Count Array Element](https://leetgpu.com/challenges/count-array-element) | Medium |
| [15 · Mean Squared Error](../problems/15_mean_squared_error/README.md) | [Mean Squared Error](https://leetgpu.com/challenges/mean-squared-error) | Medium |
| [16 · Dot Product](../problems/16_dot_product/README.md) | [Dot Product](https://leetgpu.com/challenges/dot-product) | Medium |
| [17 · Prefix Sum](../problems/17_prefix_sum/README.md) | [Prefix Sum](https://leetgpu.com/challenges/prefix-sum) | Medium |
| [18 · Histogramming](../problems/18_histogramming/README.md) | [Histogramming](https://leetgpu.com/challenges/histogramming) | Medium |
| [19 · Stream Compaction](../problems/19_stream_compaction/README.md) | [Stream Compaction](https://leetgpu.com/challenges/stream-compaction) | Medium |
| [20 · Parallel Merge](../problems/20_parallel_merge/README.md) | [Parallel Merge](https://leetgpu.com/challenges/parallel-merge) | Medium |
| [22 · Sparse Matrix-Vector Multiplication](../problems/22_sparse_matvec/README.md) | [Sparse Matrix-Vector Multiplication](https://leetgpu.com/challenges/sparse-matrix-vector-multiplication) | Medium |
| [23 · Matrix Multiplication](../problems/23_matrix_multiplication_fp32/README.md) | [Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication) | Easy |
| [24 · Batched Matrix Multiplication](../problems/24_batched_matmul_fp32/README.md) | [Batched Matrix Multiplication](https://leetgpu.com/challenges/batched-matrix-multiplication) | Medium |
| [25 · General Matrix Multiplication (GEMM)](../problems/25_gemm_fp16/README.md) | [General Matrix Multiplication (GEMM)](https://leetgpu.com/challenges/general-matrix-multiplication-gemm) | Medium |
| [26 · FP16 Batched Matrix Multiplication](../problems/26_batched_matmul_fp16/README.md) | [FP16 Batched Matrix Multiplication](https://leetgpu.com/challenges/fp16-batched-matrix-multiplication) | Medium |
| [27 · INT8 Quantized MatMul](../problems/27_int8_quantized_matmul/README.md) | [INT8 Quantized MatMul](https://leetgpu.com/challenges/int8-quantized-matmul) | Medium |
| [28 · Sparse Matrix-Dense Matrix Multiplication](../problems/28_sparse_dense_matmul/README.md) | [Sparse Matrix-Dense Matrix Multiplication](https://leetgpu.com/challenges/sparse-matrix-dense-matrix-multiplication) | Medium |
| [30 · Sigmoid Linear Unit](../problems/30_silu/README.md) | [Sigmoid Linear Unit](https://leetgpu.com/challenges/sigmoid-linear-unit) | Easy |
| [31 · Swish-Gated Linear Unit](../problems/31_swiglu_activation/README.md) | [Swish-Gated Linear Unit](https://leetgpu.com/challenges/swish-gated-linear-unit) | Easy |
| [32 · Softmax](../problems/32_softmax/README.md) | [Softmax](https://leetgpu.com/challenges/softmax) | Medium |
| [33 · RMS Normalization](../problems/33_rms_norm/README.md) | [RMS Normalization](https://leetgpu.com/challenges/rms-normalization) | Medium |
| [34 · Layer Normalization](../problems/34_layer_norm/README.md) | [Layer Normalization](https://leetgpu.com/challenges/layer-normalization) | Medium |
| [35 · Batch Normalization](../problems/35_batch_norm/README.md) | [Batch Normalization](https://leetgpu.com/challenges/batch-normalization) | Medium |
| [36 · Fused Residual Add and RMS Norm](../problems/36_fused_residual_rms_norm/README.md) | [Fused Residual Add and RMS Norm](https://leetgpu.com/challenges/fused-residual-add-and-rms-norm) | Medium |
| [37 · Categorical Cross Entropy Loss](../problems/37_categorical_cross_entropy/README.md) | [Categorical Cross Entropy Loss](https://leetgpu.com/challenges/categorical-cross-entropy-loss) | Medium |
| [38 · Rotary Positional Embedding](../problems/38_rope/README.md) | [Rotary Positional Embedding](https://leetgpu.com/challenges/rotary-positional-embedding) | Medium |
| [39 · SwiGLU MLP Block](../problems/39_swiglu_mlp/README.md) | [SwiGLU MLP Block](https://leetgpu.com/challenges/swiglu-mlp-block) | Medium |
| [40 · Softmax Attention](../problems/40_softmax_attention/README.md) | [Softmax Attention](https://leetgpu.com/challenges/softmax-attention) | Medium |
| [41 · Causal Self-Attention](../problems/41_causal_attention/README.md) | [Causal Self-Attention](https://leetgpu.com/challenges/causal-self-attention) | Hard |
| [42 · Multi-Head Attention](../problems/42_multi_head_attention/README.md) | [Multi-Head Attention](https://leetgpu.com/challenges/multi-head-attention) | Hard |
| [43 · Attention with Linear Biases](../problems/43_alibi_attention/README.md) | [Attention with Linear Biases](https://leetgpu.com/challenges/attention-with-linear-biases) | Medium |
| [44 · Grouped Query Attention](../problems/44_grouped_query_attention/README.md) | [Grouped Query Attention](https://leetgpu.com/challenges/grouped-query-attention) | Medium |
| [45 · Decaying Causal Attention](../problems/45_decaying_causal_attention/README.md) | [Decaying Causal Attention](https://leetgpu.com/challenges/decaying-causal-attention) | Medium |

## Portfolio Extensions

These seven entries have independently specified scope and no claimed LeetGPU URL:

- [21 · Dense GEMV](../problems/21_dense_gemv/README.md) — Low-arithmetic-intensity matrix math.
- [29 · Fused GEMM + Bias + Activation](../problems/29_fused_gemm_epilogue/README.md) — GEMM epilogue fusion.
- [46 · Quantize / Dequantize Pipeline](../problems/46_quantization_pipeline/README.md) — Quantization error vs bandwidth.
- [47 · KV-Cache Update and Paged Access](../problems/47_kv_cache_operations/README.md) — Stateful cache layout.
- [48 · FlashAttention-Style Online Attention](../problems/48_flash_attention/README.md) — Numerically stable online reduction.
- [49 · Paged Attention](../problems/49_paged_attention/README.md) — Irregular decode-time attention.
- [50 · MoE Token Routing and Dispatch](../problems/50_moe_token_routing/README.md) — Balanced scatter and expert dispatch.

Dense GEMV fills a missing prerequisite before dense GEMM; the fused GEMM
extension studies an inference epilogue rather than another multiplication
alias. Quantization computes scales and conversion, beyond the source that
receives scales. KV-cache work introduces mutation and pages; online and paged
attention test different memory/scheduling models. MoE dispatch starts from
assignments and goes beyond selecting experts.

## Deliberately excluded or deferred

These were visible catalog entries, not claims that the topics are unimportant.
They remain possible later electives, outside the 50 core totals.

| Visible catalog entries | Why not separate core studies now? |
|---|---|
| [Matrix Addition](https://leetgpu.com/challenges/matrix-addition), [Color Inversion](https://leetgpu.com/challenges/color-inversion), [Value Clipping](https://leetgpu.com/challenges/value-clipping), [Leaky ReLU](https://leetgpu.com/challenges/leaky-relu) | Too much overlap with vector mapping, ReLU, and the existing layout studies. |
| [Sigmoid Activation](https://leetgpu.com/challenges/sigmoid-activation), [Gaussian Error Gated Linear Unit](https://leetgpu.com/challenges/gaussian-error-gated-linear-unit) | SiLU and SwiGLU already provide special-function and gating lessons; keep approximation variants within those studies first. |
| [Count 2D Array Element](https://leetgpu.com/challenges/count-2d-array-element), [Count 3D Array Element](https://leetgpu.com/challenges/count-3d-array-element) | Rank changes alone do not justify separate counting studies. |
| [Subarray Sum](https://leetgpu.com/challenges/subarray-sum), [2D Subarray Sum](https://leetgpu.com/challenges/2d-subarray-sum), [3D Subarray Sum](https://leetgpu.com/challenges/3d-subarray-sum), [Max Subarray Sum](https://leetgpu.com/challenges/max-subarray-sum) | Reduction, prefixes, and window locality provide the core prerequisites; region-query algorithms can be a later branch. |
| [2D Convolution](https://leetgpu.com/challenges/2d-convolution), [3D Convolution](https://leetgpu.com/challenges/3d-convolution), [2D Jacobi Stencil](https://leetgpu.com/challenges/2d-jacobi-stencil) | Avoid a stencil-heavy curriculum. Keep 1D halo reuse and Gaussian separability as distinct experiments. |
| [Causal Depthwise Conv1d](https://leetgpu.com/challenges/causal-depthwise-conv1d), [Group Normalization](https://leetgpu.com/challenges/group-normalization) | Good model-specific electives, but the first core already covers stencil and normalization mechanisms. |
| [FP16 Dot Product](https://leetgpu.com/challenges/fp16-dot-product), [Matrix Power](https://leetgpu.com/challenges/matrix-power) | Treat precision as a labeled dot-product variant; repeated GEMMs add less than the selected scheduling and epilogue studies. |
| [Top K Selection](https://leetgpu.com/challenges/top-k-selection), [Top-p Sampling](https://leetgpu.com/challenges/top-p-sampling), [MoE Top-K Gating](https://leetgpu.com/challenges/moe-top-k-gating) | Valuable next electives. Core MoE routing begins with supplied assignments so dispatch can be studied independently; selection/sampling can then become separate projects. |
| [Segmented Exclusive Prefix Sum](https://leetgpu.com/challenges/segmented-exclusive-prefix-sum), [Linear Recurrence](https://leetgpu.com/challenges/linear-recurrence), [SSM Selective Scan](https://leetgpu.com/challenges/ssm-selective-scan), [Parallel Reverse Scan (GAE)](https://leetgpu.com/challenges/parallel-reverse-scan-gae) | Strong second-wave progression after ordinary scan and decaying causal attention; avoid introducing several recurrence families simultaneously. |
| [INT4 Weight-Only Quantized MatMul](https://leetgpu.com/challenges/int4-weight-only-quantized-matmul), [INT8 KV-Cache Attention](https://leetgpu.com/challenges/int8-kv-cache-attention) | Combine quantization and inference only after their separate contracts are validated. These are high-value future extensions. |
| [Softmax Attention Backward](https://leetgpu.com/challenges/softmax-attention-backward) | Important training-oriented follow-up, but forward correctness, online state, and resource reasoning come first. |
| [Multi-Head Cross-Attention](https://leetgpu.com/challenges/multi-head-cross-attention), [Sliding Window Self-Attention](https://leetgpu.com/challenges/sliding-window-self-attention), [Attention with Sinks](https://leetgpu.com/challenges/attention-with-sinks), [Linear Self-Attention](https://leetgpu.com/challenges/linear-self-attention) | Limit mask/attention-family duplication. The retained path already isolates causality, heads, positional bias, K/V sharing, recurrence, online normalization, and paging. |
| [GPT-2 Transformer Block](https://leetgpu.com/challenges/gpt-2-transformer-block), [Llama Transformer Block](https://leetgpu.com/challenges/llama-transformer-block), [Diffusion Transformer Block](https://leetgpu.com/challenges/diffusion-transformer-block), [Adder Transformer Inference](https://leetgpu.com/challenges/adder-transformer-inference), [Simple Inference](https://leetgpu.com/challenges/simple-inference) | Whole-model orchestration can hide which kernel change explains the gain; compose validated primitives later. |
| [Token Embedding Layer](https://leetgpu.com/challenges/token-embedding-layer), [LoRA Linear](https://leetgpu.com/challenges/lora-linear), [Speculative Decoding Verification](https://leetgpu.com/challenges/speculative-decoding-verification) | Useful inference branches after layout, matrix, and cache contracts; excluded to keep the first core bounded. |
| [PPO Clipped Surrogate Loss](https://leetgpu.com/challenges/ppo-clipped-surrogate-loss), [DPO Sequence Loss](https://leetgpu.com/challenges/dpo-sequence-loss), [GRPO Surrogate Loss](https://leetgpu.com/challenges/grpo-surrogate-loss) | MSE and categorical cross entropy establish fused loss methodology first; these introduce additional algorithm-specific semantics. |
| [Ordinary Least Squares](https://leetgpu.com/challenges/ordinary-least-squares), [Logistic Regression](https://leetgpu.com/challenges/logistic-regression), [Monte Carlo Integration](https://leetgpu.com/challenges/monte-carlo-integration), [K-Means Clustering](https://leetgpu.com/challenges/k-means-clustering), [Nearest Neighbor](https://leetgpu.com/challenges/nearest-neighbor) | Broader numerical/application work rather than the most direct next low-level kernel mechanisms. |
| [Sorting](https://leetgpu.com/challenges/sorting), [Radix Sort](https://leetgpu.com/challenges/radix-sort) | Merge, scan, histogram, and compaction are the chosen bounded primitive path; a sorting study is worthwhile after that foundation. |
| [Fast Fourier Transform](https://leetgpu.com/challenges/fast-fourier-transform), [2D FFT](https://leetgpu.com/challenges/2d-fft), [BFS Shortest Path](https://leetgpu.com/challenges/bfs-shortest-path), [All-Pairs Shortest Paths](https://leetgpu.com/challenges/all-pairs-shortest-paths), [Multi-Agent Simulation](https://leetgpu.com/challenges/multi-agent-simulation) | Strong HPC electives, but advanced core slots favor transformer/inference memory behavior. |

## Why the retained overlaps earn their slots

Matrix Copy is the traffic control for Matrix Transpose, not another output
formula. FP32 batching tests scheduling; FP16 GEMM introduces matrix instructions;
FP16 batching then tests their granularity under small workloads. SiLU isolates
special-function cost, the gated activation removes pointwise intermediates,
and the full SwiGLU MLP studies fusion boundaries around matrix multiplication.
These are different hypotheses, not extra counters on a solved-problem list.
