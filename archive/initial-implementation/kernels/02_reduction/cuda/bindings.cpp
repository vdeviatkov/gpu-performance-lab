#include <torch/extension.h>

at::Tensor reduction_naive(const at::Tensor &);
at::Tensor reduction_optimized(const at::Tensor &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("naive", &reduction_naive, "reduction baseline (CUDA, forward only)");
  m.def("optimized", &reduction_optimized, "reduction optimized (CUDA, forward only)");
}
