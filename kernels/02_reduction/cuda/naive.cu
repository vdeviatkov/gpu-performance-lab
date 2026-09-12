#include "checks.cuh"

namespace {
constexpr int BLOCK = 256;

template <typename T>
__global__ void shared_tree(const T *__restrict__ x, float *__restrict__ partial, int64_t n) {
  __shared__ float scratch[BLOCK];
  float value = 0.0f;
  for (int64_t i = int64_t(blockIdx.x) * BLOCK + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * BLOCK)
    value += float(x[i]);
  scratch[threadIdx.x] = value;
  __syncthreads();
  // Contiguous addressing avoids the strided bank conflicts of an interleaved
  // tree, but still requires a block barrier at each level.
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    partial[blockIdx.x] = scratch[0];
}
} // namespace

at::Tensor reduction_naive(const at::Tensor &x) {
  check_input(x, 1);
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({}, x.options().dtype(at::kFloat));
  if (!x.numel())
    return out.zero_();
  const int blocks = int(std::min<int64_t>((x.numel() + BLOCK - 1) / BLOCK, 4096));
  auto partial = at::empty({blocks}, out.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "shared_tree", [&] {
    shared_tree<scalar_t><<<blocks, BLOCK, 0, stream>>>(x.data_ptr<scalar_t>(),
                                                        partial.data_ptr<float>(), x.numel());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  shared_tree<float>
      <<<1, BLOCK, 0, stream>>>(partial.data_ptr<float>(), out.data_ptr<float>(), blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
