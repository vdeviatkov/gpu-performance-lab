"""Public contract: contiguous, inference-only tensors and explicit precision."""


def validate(x, *, ndim, cuda=False, other=None):
    import torch

    if x.ndim != ndim:
        raise ValueError(f"Expected {ndim} dimensions, got {x.ndim}")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("Supported dtypes: float32, float16, bfloat16")
    if not x.is_contiguous():
        raise ValueError("Input must be contiguous; copies are not hidden inside kernels")
    if x.requires_grad:
        raise ValueError("These forward-only experiments do not implement autograd")
    if cuda and x.device.type != "cuda":
        raise ValueError("Expected a CUDA tensor")
    if ndim == 2 and x.shape[1] == 0:
        raise ValueError("Softmax requires a nonempty last dimension")
    if other is not None:
        validate(other, ndim=ndim, cuda=cuda)
        if x.shape != other.shape or x.dtype != other.dtype or x.device != other.device:
            raise ValueError("Inputs must have identical shape, dtype, and device")
