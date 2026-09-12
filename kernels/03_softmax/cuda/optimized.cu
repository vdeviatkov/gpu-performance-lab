#include "checks.cuh"
#include "reduce.cuh"

namespace {
constexpr int BLOCK = 256;

template <typename T>
__global__ void block_row(const T *__restrict__ x, T *__restrict__ out, int64_t rows,
                          int64_t cols) {
  __shared__ float scratch[BLOCK / 32];
  __shared__ float row_max;
  __shared__ float row_sum;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t base = row * cols;
    float maximum = -CUDART_INF_F;
    for (int64_t col = threadIdx.x; col < cols; col += BLOCK)
      maximum = fmaxf(maximum, float(x[base + col]));
    maximum = block_max<BLOCK>(maximum, scratch);
    if (threadIdx.x == 0)
      row_max = maximum;
    __syncthreads(); // Publish max and ensure scratch is safe to reuse.
    float sum = 0.0f;
    for (int64_t col = threadIdx.x; col < cols; col += BLOCK)
      sum += expf(float(x[base + col]) - row_max);
    sum = block_sum<BLOCK>(sum, scratch);
    if (threadIdx.x == 0)
      row_sum = sum;
    __syncthreads();
    for (int64_t col = threadIdx.x; col < cols; col += BLOCK)
      out[base + col] = T(expf(float(x[base + col]) - row_max) / row_sum);
    __syncthreads(); // Protect row_max/row_sum before this block takes another row.
  }
}
} // namespace

at::Tensor softmax_optimized(const at::Tensor &x) {
  check_input(x, 2);
  TORCH_CHECK(x.size(1) > 0, "Softmax requires nonempty rows");
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty_like(x);
  if (!x.size(0))
    return out;
  const int blocks = int(std::min<int64_t>(x.size(0), 65535));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "block_row", [&] {
    block_row<<<blocks, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), x.size(0), x.size(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
