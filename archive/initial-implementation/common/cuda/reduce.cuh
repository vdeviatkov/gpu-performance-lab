#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

// All 32 lanes execute every shuffle, including lanes holding identity values.
__device__ inline float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

__device__ inline float warp_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  return value;
}

// BLOCK must be a multiple of 32. Only lane 0 of each warp writes shared memory.
// The returned aggregate is valid in thread 0 only.
template <int BLOCK> __device__ float block_sum(float value, float *scratch) {
  static_assert(BLOCK % 32 == 0 && BLOCK <= 1024);
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  value = warp_sum(value);
  if (lane == 0)
    scratch[warp] = value;
  __syncthreads();
  value = threadIdx.x < BLOCK / 32 ? scratch[lane] : 0.0f;
  if (warp == 0)
    value = warp_sum(value);
  return value;
}

template <int BLOCK> __device__ float block_max(float value, float *scratch) {
  static_assert(BLOCK % 32 == 0 && BLOCK <= 1024);
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  value = warp_max(value);
  if (lane == 0)
    scratch[warp] = value;
  __syncthreads();
  value = threadIdx.x < BLOCK / 32 ? scratch[lane] : -CUDART_INF_F;
  if (warp == 0)
    value = warp_max(value);
  return value;
}
