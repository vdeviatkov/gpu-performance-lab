#include <torch/extension.h>

at::Tensor vector_add_naive(const at::Tensor &, const at::Tensor &);
at::Tensor vector_add_optimized(const at::Tensor &, const at::Tensor &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("naive", &vector_add_naive, "vector_add baseline (CUDA, forward only)");
  m.def("optimized", &vector_add_optimized, "vector_add optimized (CUDA, forward only)");
}
