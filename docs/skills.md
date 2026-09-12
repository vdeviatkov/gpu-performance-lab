# Skill map

Links cross-reference the ordered curriculum; a study can appear in several groups.
This is a manually maintainable view, not a completion claim.

## Indexing

- [01 · Vector Addition](../problems/01_vector_add/README.md) — **P1**; primary: Launch overhead vs bandwidth
- [02 · ReLU](../problems/02_relu/README.md) — **P2**; primary: Predication and branching
- [03 · Reverse Array](../problems/03_reverse_array/README.md) — **P2**; primary: Race-free in-place indexing
- [04 · Interleave Arrays](../problems/04_interleave_arrays/README.md) — **P2**; primary: Lane-to-output mapping

## Memory coalescing

- [01 · Vector Addition](../problems/01_vector_add/README.md) — **P1**; primary: Launch overhead vs bandwidth
- [03 · Reverse Array](../problems/03_reverse_array/README.md) — **P2**; primary: Race-free in-place indexing
- [04 · Interleave Arrays](../problems/04_interleave_arrays/README.md) — **P2**; primary: Lane-to-output mapping
- [05 · RGB to Grayscale](../problems/05_rgb_to_grayscale/README.md) — **P2**; primary: Interleaved channel access
- [07 · Matrix Copy](../problems/07_matrix_copy/README.md) — **P1**; primary: Memory bandwidth ceiling
- [08 · Matrix Transpose](../problems/08_matrix_transpose/README.md) — **P0**; primary: Shared-memory bank conflicts
- [21 · Dense GEMV](../problems/21_dense_gemv/README.md) — **P1**; primary: Low-arithmetic-intensity matrix math
- [35 · Batch Normalization](../problems/35_batch_norm/README.md) — **P1**; primary: Reduction-axis layout
- [38 · Rotary Positional Embedding](../problems/38_rope/README.md) — **P1**; primary: Positional pair layouts

## Vectorized access

- [01 · Vector Addition](../problems/01_vector_add/README.md) — **P1**; primary: Launch overhead vs bandwidth
- [05 · RGB to Grayscale](../problems/05_rgb_to_grayscale/README.md) — **P2**; primary: Interleaved channel access
- [07 · Matrix Copy](../problems/07_matrix_copy/README.md) — **P1**; primary: Memory bandwidth ceiling
- [38 · Rotary Positional Embedding](../problems/38_rope/README.md) — **P1**; primary: Positional pair layouts

## Launch overhead

- [01 · Vector Addition](../problems/01_vector_add/README.md) — **P1**; primary: Launch overhead vs bandwidth
- [24 · Batched Matrix Multiplication](../problems/24_batched_matmul_fp32/README.md) — **P0**; primary: Batch scheduling and occupancy

## Branching and predication

- [02 · ReLU](../problems/02_relu/README.md) — **P2**; primary: Predication and branching
- [14 · Count Array Element](../problems/14_count_array_element/README.md) — **P1**; primary: Predicate reduction

## Instruction mix

- [02 · ReLU](../problems/02_relu/README.md) — **P2**; primary: Predication and branching
- [06 · Rainbow Table](../problems/06_rainbow_table/README.md) — **P2**; primary: Integer instruction throughput
- [16 · Dot Product](../problems/16_dot_product/README.md) — **P1**; primary: Fused multiply-accumulate reduction
- [30 · Sigmoid Linear Unit](../problems/30_silu/README.md) — **P2**; primary: Special-function throughput
- [31 · Swish-Gated Linear Unit](../problems/31_swiglu_activation/README.md) — **P1**; primary: Elementwise gating fusion

## In-place ownership

- [03 · Reverse Array](../problems/03_reverse_array/README.md) — **P2**; primary: Race-free in-place indexing

## Data layouts

- [04 · Interleave Arrays](../problems/04_interleave_arrays/README.md) — **P2**; primary: Lane-to-output mapping
- [05 · RGB to Grayscale](../problems/05_rgb_to_grayscale/README.md) — **P2**; primary: Interleaved channel access
- [12 · Weight Dequantization](../problems/12_weight_dequantization/README.md) — **P1**; primary: Scale-tile locality
- [21 · Dense GEMV](../problems/21_dense_gemv/README.md) — **P1**; primary: Low-arithmetic-intensity matrix math
- [31 · Swish-Gated Linear Unit](../problems/31_swiglu_activation/README.md) — **P1**; primary: Elementwise gating fusion
- [35 · Batch Normalization](../problems/35_batch_norm/README.md) — **P1**; primary: Reduction-axis layout
- [38 · Rotary Positional Embedding](../problems/38_rope/README.md) — **P1**; primary: Positional pair layouts
- [42 · Multi-Head Attention](../problems/42_multi_head_attention/README.md) — **P0**; primary: Head layout and scheduling
- [44 · Grouped Query Attention](../problems/44_grouped_query_attention/README.md) — **P0**; primary: Shared K/V head reuse
- [47 · KV-Cache Update and Paged Access](../problems/47_kv_cache_operations/README.md) — **P0**; primary: Stateful cache layout

## Dependency chains

- [06 · Rainbow Table](../problems/06_rainbow_table/README.md) — **P2**; primary: Integer instruction throughput
- [16 · Dot Product](../problems/16_dot_product/README.md) — **P1**; primary: Fused multiply-accumulate reduction

## Occupancy and registers

- [06 · Rainbow Table](../problems/06_rainbow_table/README.md) — **P2**; primary: Integer instruction throughput
- [24 · Batched Matrix Multiplication](../problems/24_batched_matmul_fp32/README.md) — **P0**; primary: Batch scheduling and occupancy
- [32 · Softmax](../problems/32_softmax/README.md) — **P0**; primary: Numerically stable fused reduction
- [36 · Fused Residual Add and RMS Norm](../problems/36_fused_residual_rms_norm/README.md) — **P0**; primary: Residual-normalization fusion
- [39 · SwiGLU MLP Block](../problems/39_swiglu_mlp/README.md) — **P0**; primary: MLP fusion boundaries

## Cache behavior

- [07 · Matrix Copy](../problems/07_matrix_copy/README.md) — **P1**; primary: Memory bandwidth ceiling
- [11 · 2D Max Pooling](../problems/11_max_pooling_2d/README.md) — **P2**; primary: Overlapping window locality
- [12 · Weight Dequantization](../problems/12_weight_dequantization/README.md) — **P1**; primary: Scale-tile locality
- [44 · Grouped Query Attention](../problems/44_grouped_query_attention/README.md) — **P0**; primary: Shared K/V head reuse

## Shared memory

- [08 · Matrix Transpose](../problems/08_matrix_transpose/README.md) — **P0**; primary: Shared-memory bank conflicts
- [09 · 1D Convolution](../problems/09_convolution_1d/README.md) — **P1**; primary: Halo reuse in shared memory
- [10 · Gaussian Blur](../problems/10_gaussian_blur/README.md) — **P1**; primary: Separable filtering tradeoffs
- [13 · Reduction](../problems/13_reduction/README.md) — **P0**; primary: Warp-level reduction
- [17 · Prefix Sum](../problems/17_prefix_sum/README.md) — **P1**; primary: Cross-block prefix propagation
- [18 · Histogramming](../problems/18_histogramming/README.md) — **P1**; primary: Atomic contention
- [23 · Matrix Multiplication](../problems/23_matrix_multiplication_fp32/README.md) — **P1**; primary: Shared-memory data reuse
- [25 · General Matrix Multiplication (GEMM)](../problems/25_gemm_fp16/README.md) — **P0**; primary: Tensor Core utilization
- [48 · FlashAttention-Style Online Attention](../problems/48_flash_attention/README.md) — **P0**; primary: Numerically stable online reduction

## Bank conflicts

- [08 · Matrix Transpose](../problems/08_matrix_transpose/README.md) — **P0**; primary: Shared-memory bank conflicts
- [18 · Histogramming](../problems/18_histogramming/README.md) — **P1**; primary: Atomic contention

## Tiled memory access

- [08 · Matrix Transpose](../problems/08_matrix_transpose/README.md) — **P0**; primary: Shared-memory bank conflicts
- [09 · 1D Convolution](../problems/09_convolution_1d/README.md) — **P1**; primary: Halo reuse in shared memory
- [10 · Gaussian Blur](../problems/10_gaussian_blur/README.md) — **P1**; primary: Separable filtering tradeoffs
- [11 · 2D Max Pooling](../problems/11_max_pooling_2d/README.md) — **P2**; primary: Overlapping window locality
- [23 · Matrix Multiplication](../problems/23_matrix_multiplication_fp32/README.md) — **P1**; primary: Shared-memory data reuse
- [28 · Sparse Matrix-Dense Matrix Multiplication](../problems/28_sparse_dense_matmul/README.md) — **P1**; primary: Sparse reuse across output columns
- [41 · Causal Self-Attention](../problems/41_causal_attention/README.md) — **P1**; primary: Causal masking and triangular work

## Boundary handling

- [09 · 1D Convolution](../problems/09_convolution_1d/README.md) — **P1**; primary: Halo reuse in shared memory
- [11 · 2D Max Pooling](../problems/11_max_pooling_2d/README.md) — **P2**; primary: Overlapping window locality

## Recomputation vs traffic

- [10 · Gaussian Blur](../problems/10_gaussian_blur/README.md) — **P1**; primary: Separable filtering tradeoffs
- [29 · Fused GEMM + Bias + Activation](../problems/29_fused_gemm_epilogue/README.md) — **P1**; primary: GEMM epilogue fusion
- [36 · Fused Residual Add and RMS Norm](../problems/36_fused_residual_rms_norm/README.md) — **P0**; primary: Residual-normalization fusion
- [45 · Decaying Causal Attention](../problems/45_decaying_causal_attention/README.md) — **P1**; primary: Recurrence vs quadratic materialization
- [48 · FlashAttention-Style Online Attention](../problems/48_flash_attention/README.md) — **P0**; primary: Numerically stable online reduction

## Scale metadata

- [12 · Weight Dequantization](../problems/12_weight_dequantization/README.md) — **P1**; primary: Scale-tile locality

## Reductions

- [13 · Reduction](../problems/13_reduction/README.md) — **P0**; primary: Warp-level reduction
- [14 · Count Array Element](../problems/14_count_array_element/README.md) — **P1**; primary: Predicate reduction
- [15 · Mean Squared Error](../problems/15_mean_squared_error/README.md) — **P1**; primary: Map-reduce fusion
- [16 · Dot Product](../problems/16_dot_product/README.md) — **P1**; primary: Fused multiply-accumulate reduction
- [21 · Dense GEMV](../problems/21_dense_gemv/README.md) — **P1**; primary: Low-arithmetic-intensity matrix math
- [22 · Sparse Matrix-Vector Multiplication](../problems/22_sparse_matvec/README.md) — **P1**; primary: Irregular sparse memory access
- [32 · Softmax](../problems/32_softmax/README.md) — **P0**; primary: Numerically stable fused reduction
- [33 · RMS Normalization](../problems/33_rms_norm/README.md) — **P0**; primary: Normalization traffic and precision
- [34 · Layer Normalization](../problems/34_layer_norm/README.md) — **P0**; primary: Stable variance reduction
- [35 · Batch Normalization](../problems/35_batch_norm/README.md) — **P1**; primary: Reduction-axis layout
- [36 · Fused Residual Add and RMS Norm](../problems/36_fused_residual_rms_norm/README.md) — **P0**; primary: Residual-normalization fusion
- [37 · Categorical Cross Entropy Loss](../problems/37_categorical_cross_entropy/README.md) — **P1**; primary: Fused log-sum-exp loss
- [46 · Quantize / Dequantize Pipeline](../problems/46_quantization_pipeline/README.md) — **P1**; primary: Quantization error vs bandwidth

## Warp primitives

- [13 · Reduction](../problems/13_reduction/README.md) — **P0**; primary: Warp-level reduction
- [17 · Prefix Sum](../problems/17_prefix_sum/README.md) — **P1**; primary: Cross-block prefix propagation
- [32 · Softmax](../problems/32_softmax/README.md) — **P0**; primary: Numerically stable fused reduction
- [34 · Layer Normalization](../problems/34_layer_norm/README.md) — **P0**; primary: Stable variance reduction

## Synchronization

- [13 · Reduction](../problems/13_reduction/README.md) — **P0**; primary: Warp-level reduction
- [17 · Prefix Sum](../problems/17_prefix_sum/README.md) — **P1**; primary: Cross-block prefix propagation
- [19 · Stream Compaction](../problems/19_stream_compaction/README.md) — **P1**; primary: Stable parallel filtering

## Atomics

- [14 · Count Array Element](../problems/14_count_array_element/README.md) — **P1**; primary: Predicate reduction
- [18 · Histogramming](../problems/18_histogramming/README.md) — **P1**; primary: Atomic contention
- [50 · MoE Token Routing and Dispatch](../problems/50_moe_token_routing/README.md) — **P1**; primary: Balanced scatter and expert dispatch

## Kernel fusion

- [15 · Mean Squared Error](../problems/15_mean_squared_error/README.md) — **P1**; primary: Map-reduce fusion
- [29 · Fused GEMM + Bias + Activation](../problems/29_fused_gemm_epilogue/README.md) — **P1**; primary: GEMM epilogue fusion
- [30 · Sigmoid Linear Unit](../problems/30_silu/README.md) — **P2**; primary: Special-function throughput
- [31 · Swish-Gated Linear Unit](../problems/31_swiglu_activation/README.md) — **P1**; primary: Elementwise gating fusion
- [32 · Softmax](../problems/32_softmax/README.md) — **P0**; primary: Numerically stable fused reduction
- [33 · RMS Normalization](../problems/33_rms_norm/README.md) — **P0**; primary: Normalization traffic and precision
- [34 · Layer Normalization](../problems/34_layer_norm/README.md) — **P0**; primary: Stable variance reduction
- [36 · Fused Residual Add and RMS Norm](../problems/36_fused_residual_rms_norm/README.md) — **P0**; primary: Residual-normalization fusion
- [37 · Categorical Cross Entropy Loss](../problems/37_categorical_cross_entropy/README.md) — **P1**; primary: Fused log-sum-exp loss
- [39 · SwiGLU MLP Block](../problems/39_swiglu_mlp/README.md) — **P0**; primary: MLP fusion boundaries
- [43 · Attention with Linear Biases](../problems/43_alibi_attention/README.md) — **P1**; primary: Fused attention bias generation
- [46 · Quantize / Dequantize Pipeline](../problems/46_quantization_pipeline/README.md) — **P1**; primary: Quantization error vs bandwidth
- [50 · MoE Token Routing and Dispatch](../problems/50_moe_token_routing/README.md) — **P1**; primary: Balanced scatter and expert dispatch

## Numerical precision

- [15 · Mean Squared Error](../problems/15_mean_squared_error/README.md) — **P1**; primary: Map-reduce fusion
- [16 · Dot Product](../problems/16_dot_product/README.md) — **P1**; primary: Fused multiply-accumulate reduction
- [30 · Sigmoid Linear Unit](../problems/30_silu/README.md) — **P2**; primary: Special-function throughput

## Prefix operations

- [17 · Prefix Sum](../problems/17_prefix_sum/README.md) — **P1**; primary: Cross-block prefix propagation
- [19 · Stream Compaction](../problems/19_stream_compaction/README.md) — **P1**; primary: Stable parallel filtering
- [45 · Decaying Causal Attention](../problems/45_decaying_causal_attention/README.md) — **P1**; primary: Recurrence vs quadratic materialization
- [50 · MoE Token Routing and Dispatch](../problems/50_moe_token_routing/README.md) — **P1**; primary: Balanced scatter and expert dispatch

## Data-dependent performance

- [18 · Histogramming](../problems/18_histogramming/README.md) — **P1**; primary: Atomic contention

## Irregular memory access

- [19 · Stream Compaction](../problems/19_stream_compaction/README.md) — **P1**; primary: Stable parallel filtering
- [20 · Parallel Merge](../problems/20_parallel_merge/README.md) — **P1**; primary: Balanced irregular partitioning
- [22 · Sparse Matrix-Vector Multiplication](../problems/22_sparse_matvec/README.md) — **P1**; primary: Irregular sparse memory access
- [28 · Sparse Matrix-Dense Matrix Multiplication](../problems/28_sparse_dense_matmul/README.md) — **P1**; primary: Sparse reuse across output columns
- [37 · Categorical Cross Entropy Loss](../problems/37_categorical_cross_entropy/README.md) — **P1**; primary: Fused log-sum-exp loss
- [47 · KV-Cache Update and Paged Access](../problems/47_kv_cache_operations/README.md) — **P0**; primary: Stateful cache layout
- [49 · Paged Attention](../problems/49_paged_attention/README.md) — **P0**; primary: Irregular decode-time attention
- [50 · MoE Token Routing and Dispatch](../problems/50_moe_token_routing/README.md) — **P1**; primary: Balanced scatter and expert dispatch

## Output ownership

- [19 · Stream Compaction](../problems/19_stream_compaction/README.md) — **P1**; primary: Stable parallel filtering
- [47 · KV-Cache Update and Paged Access](../problems/47_kv_cache_operations/README.md) — **P0**; primary: Stateful cache layout

## Divergence

- [20 · Parallel Merge](../problems/20_parallel_merge/README.md) — **P1**; primary: Balanced irregular partitioning

## Work partitioning

- [20 · Parallel Merge](../problems/20_parallel_merge/README.md) — **P1**; primary: Balanced irregular partitioning
- [22 · Sparse Matrix-Vector Multiplication](../problems/22_sparse_matvec/README.md) — **P1**; primary: Irregular sparse memory access
- [24 · Batched Matrix Multiplication](../problems/24_batched_matmul_fp32/README.md) — **P0**; primary: Batch scheduling and occupancy
- [26 · FP16 Batched Matrix Multiplication](../problems/26_batched_matmul_fp16/README.md) — **P1**; primary: Mixed-precision batch utilization
- [28 · Sparse Matrix-Dense Matrix Multiplication](../problems/28_sparse_dense_matmul/README.md) — **P1**; primary: Sparse reuse across output columns
- [42 · Multi-Head Attention](../problems/42_multi_head_attention/README.md) — **P0**; primary: Head layout and scheduling
- [44 · Grouped Query Attention](../problems/44_grouped_query_attention/README.md) — **P0**; primary: Shared K/V head reuse
- [49 · Paged Attention](../problems/49_paged_attention/README.md) — **P0**; primary: Irregular decode-time attention
- [50 · MoE Token Routing and Dispatch](../problems/50_moe_token_routing/README.md) — **P1**; primary: Balanced scatter and expert dispatch

## Arithmetic intensity

- [21 · Dense GEMV](../problems/21_dense_gemv/README.md) — **P1**; primary: Low-arithmetic-intensity matrix math
- [23 · Matrix Multiplication](../problems/23_matrix_multiplication_fp32/README.md) — **P1**; primary: Shared-memory data reuse

## Sparse operations

- [22 · Sparse Matrix-Vector Multiplication](../problems/22_sparse_matvec/README.md) — **P1**; primary: Irregular sparse memory access
- [28 · Sparse Matrix-Dense Matrix Multiplication](../problems/28_sparse_dense_matmul/README.md) — **P1**; primary: Sparse reuse across output columns

## GEMM

- [23 · Matrix Multiplication](../problems/23_matrix_multiplication_fp32/README.md) — **P1**; primary: Shared-memory data reuse
- [24 · Batched Matrix Multiplication](../problems/24_batched_matmul_fp32/README.md) — **P0**; primary: Batch scheduling and occupancy
- [25 · General Matrix Multiplication (GEMM)](../problems/25_gemm_fp16/README.md) — **P0**; primary: Tensor Core utilization
- [26 · FP16 Batched Matrix Multiplication](../problems/26_batched_matmul_fp16/README.md) — **P1**; primary: Mixed-precision batch utilization
- [27 · INT8 Quantized MatMul](../problems/27_int8_quantized_matmul/README.md) — **P1**; primary: Quantized arithmetic contracts
- [29 · Fused GEMM + Bias + Activation](../problems/29_fused_gemm_epilogue/README.md) — **P1**; primary: GEMM epilogue fusion
- [39 · SwiGLU MLP Block](../problems/39_swiglu_mlp/README.md) — **P0**; primary: MLP fusion boundaries
- [40 · Softmax Attention](../problems/40_softmax_attention/README.md) — **P1**; primary: Attention pipeline decomposition

## Tensor Cores

- [25 · General Matrix Multiplication (GEMM)](../problems/25_gemm_fp16/README.md) — **P0**; primary: Tensor Core utilization
- [26 · FP16 Batched Matrix Multiplication](../problems/26_batched_matmul_fp16/README.md) — **P1**; primary: Mixed-precision batch utilization
- [27 · INT8 Quantized MatMul](../problems/27_int8_quantized_matmul/README.md) — **P1**; primary: Quantized arithmetic contracts
- [48 · FlashAttention-Style Online Attention](../problems/48_flash_attention/README.md) — **P0**; primary: Numerically stable online reduction

## Mixed precision

- [25 · General Matrix Multiplication (GEMM)](../problems/25_gemm_fp16/README.md) — **P0**; primary: Tensor Core utilization
- [26 · FP16 Batched Matrix Multiplication](../problems/26_batched_matmul_fp16/README.md) — **P1**; primary: Mixed-precision batch utilization
- [27 · INT8 Quantized MatMul](../problems/27_int8_quantized_matmul/README.md) — **P1**; primary: Quantized arithmetic contracts
- [29 · Fused GEMM + Bias + Activation](../problems/29_fused_gemm_epilogue/README.md) — **P1**; primary: GEMM epilogue fusion
- [33 · RMS Normalization](../problems/33_rms_norm/README.md) — **P0**; primary: Normalization traffic and precision
- [39 · SwiGLU MLP Block](../problems/39_swiglu_mlp/README.md) — **P0**; primary: MLP fusion boundaries
- [42 · Multi-Head Attention](../problems/42_multi_head_attention/README.md) — **P0**; primary: Head layout and scheduling
- [46 · Quantize / Dequantize Pipeline](../problems/46_quantization_pipeline/README.md) — **P1**; primary: Quantization error vs bandwidth

## Asynchronous copies

- [25 · General Matrix Multiplication (GEMM)](../problems/25_gemm_fp16/README.md) — **P0**; primary: Tensor Core utilization

## Quantization

- [27 · INT8 Quantized MatMul](../problems/27_int8_quantized_matmul/README.md) — **P1**; primary: Quantized arithmetic contracts
- [46 · Quantize / Dequantize Pipeline](../problems/46_quantization_pipeline/README.md) — **P1**; primary: Quantization error vs bandwidth

## Numerical stability

- [32 · Softmax](../problems/32_softmax/README.md) — **P0**; primary: Numerically stable fused reduction
- [33 · RMS Normalization](../problems/33_rms_norm/README.md) — **P0**; primary: Normalization traffic and precision
- [34 · Layer Normalization](../problems/34_layer_norm/README.md) — **P0**; primary: Stable variance reduction
- [35 · Batch Normalization](../problems/35_batch_norm/README.md) — **P1**; primary: Reduction-axis layout
- [37 · Categorical Cross Entropy Loss](../problems/37_categorical_cross_entropy/README.md) — **P1**; primary: Fused log-sum-exp loss
- [40 · Softmax Attention](../problems/40_softmax_attention/README.md) — **P1**; primary: Attention pipeline decomposition
- [41 · Causal Self-Attention](../problems/41_causal_attention/README.md) — **P1**; primary: Causal masking and triangular work
- [45 · Decaying Causal Attention](../problems/45_decaying_causal_attention/README.md) — **P1**; primary: Recurrence vs quadratic materialization

## Positional encoding

- [38 · Rotary Positional Embedding](../problems/38_rope/README.md) — **P1**; primary: Positional pair layouts
- [43 · Attention with Linear Biases](../problems/43_alibi_attention/README.md) — **P1**; primary: Fused attention bias generation

## Attention

- [40 · Softmax Attention](../problems/40_softmax_attention/README.md) — **P1**; primary: Attention pipeline decomposition
- [41 · Causal Self-Attention](../problems/41_causal_attention/README.md) — **P1**; primary: Causal masking and triangular work
- [42 · Multi-Head Attention](../problems/42_multi_head_attention/README.md) — **P0**; primary: Head layout and scheduling
- [43 · Attention with Linear Biases](../problems/43_alibi_attention/README.md) — **P1**; primary: Fused attention bias generation
- [44 · Grouped Query Attention](../problems/44_grouped_query_attention/README.md) — **P0**; primary: Shared K/V head reuse
- [45 · Decaying Causal Attention](../problems/45_decaying_causal_attention/README.md) — **P1**; primary: Recurrence vs quadratic materialization
- [48 · FlashAttention-Style Online Attention](../problems/48_flash_attention/README.md) — **P0**; primary: Numerically stable online reduction
- [49 · Paged Attention](../problems/49_paged_attention/README.md) — **P0**; primary: Irregular decode-time attention

## Memory traffic

- [40 · Softmax Attention](../problems/40_softmax_attention/README.md) — **P1**; primary: Attention pipeline decomposition

## Masking

- [41 · Causal Self-Attention](../problems/41_causal_attention/README.md) — **P1**; primary: Causal masking and triangular work
- [43 · Attention with Linear Biases](../problems/43_alibi_attention/README.md) — **P1**; primary: Fused attention bias generation

## KV cache

- [47 · KV-Cache Update and Paged Access](../problems/47_kv_cache_operations/README.md) — **P0**; primary: Stateful cache layout
- [49 · Paged Attention](../problems/49_paged_attention/README.md) — **P0**; primary: Irregular decode-time attention

## Online reductions

- [48 · FlashAttention-Style Online Attention](../problems/48_flash_attention/README.md) — **P0**; primary: Numerically stable online reduction
- [49 · Paged Attention](../problems/49_paged_attention/README.md) — **P0**; primary: Irregular decode-time attention
