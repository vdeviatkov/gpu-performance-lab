#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

template <typename T> __global__ void scalar_coalesced(const T *a, const T *b, T *out, int64_t n) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = static_cast<T>(static_cast<float>(a[i]) + static_cast<float>(b[i]));
}

template <typename T> __global__ void grid_stride(const T *a, const T *b, T *out, int64_t n) {
  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
    out[i] = static_cast<T>(static_cast<float>(a[i]) + static_cast<float>(b[i]));
}

// Scalar lanes already coalesce. float4 aims to reduce memory/address instructions,
// not logical bytes. The wrapper verifies all three 16-byte alignments.
__global__ void aligned_float4(const float *a, const float *b, float *out, int64_t n) {
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < n / 4; i += stride) {
    const float4 x = reinterpret_cast<const float4 *>(a)[i];
    const float4 y = reinterpret_cast<const float4 *>(b)[i];
    reinterpret_cast<float4 *>(out)[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
  }
  // At most three elements, handled once, in the same launch as the bulk work.
  const int64_t tail = (n / 4) * 4 + tid;
  if (tail < n)
    out[tail] = a[tail] + b[tail];
}

void launch_add(const at::Tensor &a, const at::Tensor &b, at::Tensor &out, int variant,
                int threads) {
  const c10::cuda::CUDAGuard guard(a.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.get_device());
  const int64_t n = a.numel();
  const auto addresses = reinterpret_cast<uintptr_t>(a.data_ptr()) |
                         reinterpret_cast<uintptr_t>(b.data_ptr()) |
                         reinterpret_cast<uintptr_t>(out.data_ptr());
  if (variant == 2 && a.scalar_type() == at::kFloat && (addresses % 16 == 0)) {
    const int blocks =
        static_cast<int>(std::min<int64_t>((n + 4 * threads - 1) / (4 * threads), 4096));
    aligned_float4<<<blocks, threads, 0, stream.stream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
  } else {
    const int64_t needed = (n + threads - 1) / threads;
    TORCH_CHECK(needed <= 2147483647LL, "vector exceeds scalar launch grid limit");
    const int blocks = static_cast<int>(variant == 0 ? needed : std::min<int64_t>(needed, 4096));
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, a.scalar_type(), "vector_add", [&] {
      if (variant == 0)
        scalar_coalesced<<<blocks, threads, 0, stream.stream()>>>(
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
      else
        grid_stride<<<blocks, threads, 0, stream.stream()>>>(
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
    });
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
