#include <torch/extension.h>

at::Tensor softmax_naive(const at::Tensor &);
at::Tensor softmax_optimized(const at::Tensor &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("naive", &softmax_naive, "softmax baseline (CUDA, forward only)");
  m.def("optimized", &softmax_optimized, "softmax optimized (CUDA, forward only)");
}
