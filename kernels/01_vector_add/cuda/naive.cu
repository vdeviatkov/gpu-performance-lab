#include "checks.cuh"

template <typename T>
__global__ void add_scalar(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out,
                           int64_t n) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(blockDim.x) * gridDim.x)
    out[i] = T(float(a[i]) + float(b[i]));
}

at::Tensor vector_add_naive(const at::Tensor &a, const at::Tensor &b) {
  check_pair(a, b);
  const c10::cuda::CUDAGuard guard(a.device());
  auto out = at::empty_like(a);
  const auto n = a.numel();
  if (!n)
    return out;
  const int blocks = int(std::min<int64_t>((n + 255) / 256, 65535));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, a.scalar_type(), "add_scalar", [&] {
    add_scalar<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
