#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <math_constants.h>

inline void check_input(const at::Tensor &x, int64_t ndim) {
  TORCH_CHECK(x.is_cuda(), "Expected CUDA input");
  TORCH_CHECK(x.is_contiguous(), "Expected contiguous input");
  TORCH_CHECK(x.dim() == ndim, "Unexpected tensor rank");
  TORCH_CHECK(!x.requires_grad(), "Forward-only kernels do not support autograd");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                  x.scalar_type() == at::kBFloat16,
              "Expected FP32, FP16, or BF16");
}

inline void check_pair(const at::Tensor &a, const at::Tensor &b) {
  check_input(a, 1);
  check_input(b, 1);
  TORCH_CHECK(a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type() &&
                  a.device() == b.device(),
              "Inputs must have identical shape, dtype, and device");
}
