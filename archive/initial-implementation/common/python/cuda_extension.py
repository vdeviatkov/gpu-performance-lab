"""Lazy JIT build, outside timed regions; CMake-built modules work as well."""

import importlib
from functools import cache
from pathlib import Path

from common.python.registry import WORKLOADS, BackendUnavailable


@cache
def load_extension(workload):
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if not torch.cuda.is_available():
        raise BackendUnavailable("CUDA runtime/GPU unavailable")
    name = f"gpu_lab_{workload}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
    if CUDA_HOME is None:
        raise BackendUnavailable("CUDA toolkit/nvcc required for the first extension build")
    root = Path(__file__).resolve().parents[2]
    source = root.joinpath(*WORKLOADS[workload].split("."), "cuda")
    return load(
        name=name,
        sources=[str(source / s) for s in ("bindings.cpp", "naive.cu", "optimized.cu")],
        extra_include_paths=[str(root / "common" / "cuda")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"],
        verbose=False,
    )
