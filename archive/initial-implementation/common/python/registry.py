"""Keep workload discovery explicit and independent of optional GPU imports."""

from importlib import import_module

WORKLOADS = {
    "vector_add": "kernels.01_vector_add",
    "reduction": "kernels.02_reduction",
    "softmax": "kernels.03_softmax",
}
IMPLEMENTATIONS = ("pytorch", "jax", "triton", "cuda_naive", "cuda_optimized")


class BackendUnavailable(RuntimeError):
    """An optional backend or documented hardware capability is missing."""


def implementation(workload, backend):
    if workload not in WORKLOADS or backend not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown workload/backend: {workload}/{backend}")
    module = "cuda_impl" if backend.startswith("cuda_") else f"{backend}_impl"
    try:
        loaded = import_module(f"{WORKLOADS[workload]}.{module}")
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "triton", "jax", "jaxlib"}:
            raise BackendUnavailable(f"Install optional dependency: {exc.name}") from exc
        raise
    return getattr(loaded, backend.removeprefix("cuda_") if module == "cuda_impl" else "run")
