#include "checks.cuh"

namespace {
// One thread per row exposes serial reduction and strided accesses across a
// warp. Stable numerics are retained so the baseline isolates execution mapping.
template <typename T>
__global__ void serial_row(const T *__restrict__ x, T *__restrict__ out, int64_t rows,
                           int64_t cols) {
  for (int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; row < rows;
       row += int64_t(blockDim.x) * gridDim.x) {
    const int64_t base = row * cols;
    float maximum = -CUDART_INF_F;
    for (int64_t col = 0; col < cols; ++col)
      maximum = fmaxf(maximum, float(x[base + col]));
    float sum = 0.0f;
    for (int64_t col = 0; col < cols; ++col)
      sum += expf(float(x[base + col]) - maximum);
    for (int64_t col = 0; col < cols; ++col)
      out[base + col] = T(expf(float(x[base + col]) - maximum) / sum);
  }
}
} // namespace

at::Tensor softmax_naive(const at::Tensor &x) {
  check_input(x, 2);
  TORCH_CHECK(x.size(1) > 0, "Softmax requires nonempty rows");
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty_like(x);
  if (!x.size(0))
    return out;
  const int blocks = int(std::min<int64_t>((x.size(0) + 127) / 128, 65535));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "serial_row", [&] {
    serial_row<<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), x.size(0), x.size(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
