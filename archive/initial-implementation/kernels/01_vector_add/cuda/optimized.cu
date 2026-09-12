#include "checks.cuh"
#include <cstdint>

at::Tensor vector_add_naive(const at::Tensor &, const at::Tensor &);

// Vectorization reduces address/load instruction count; the scalar baseline is
// already coalesced. This does not reduce bytes transferred or guarantee a win.
__global__ void add_float4(const float *__restrict__ a, const float *__restrict__ b,
                           float *__restrict__ out, int64_t n) {
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < n / 4; i += stride) {
    const float4 av = reinterpret_cast<const float4 *>(a)[i];
    const float4 bv = reinterpret_cast<const float4 *>(b)[i];
    reinterpret_cast<float4 *>(out)[i] =
        make_float4(av.x + bv.x, av.y + bv.y, av.z + bv.z, av.w + bv.w);
  }
  // At most three scalar tail elements, disjoint from all vector stores.
  if (tid < n % 4) {
    const int64_t i = (n / 4) * 4 + tid;
    out[i] = a[i] + b[i];
  }
}

at::Tensor vector_add_optimized(const at::Tensor &a, const at::Tensor &b) {
  check_pair(a, b);
  const c10::cuda::CUDAGuard guard(a.device());
  // Contiguous slices may have an unaligned storage offset. Fall back safely.
  if (a.scalar_type() != at::kFloat || a.numel() < 4 ||
      reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 ||
      reinterpret_cast<uintptr_t>(b.data_ptr()) % 16)
    return vector_add_naive(a, b);
  auto out = at::empty_like(a);
  const int blocks = int(std::min<int64_t>((a.numel() / 4 + 255) / 256, 65535));
  add_float4<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), a.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
