"""Run correctness-gated, steady-state measurements and emit versioned JSON."""

import argparse
import json
import math
import random
import sys
import traceback
from pathlib import Path

from common.benchmark.metrics import derived_metrics, work_model
from common.benchmark.timing import measure
from common.correctness.checks import assert_correct
from common.python.environment import collect
from common.python.registry import IMPLEMENTATIONS, WORKLOADS, BackendUnavailable, implementation

CONFIG_DIR = Path(__file__).with_name("configs")
DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
VARIANTS = {
    "vector_add": {"cuda_naive": "scalar_coalesced", "cuda_optimized": "float4_or_scalar_fallback"},
    "reduction": {"cuda_naive": "shared_tree", "cuda_optimized": "warp_shuffle"},
    "softmax": {"cuda_naive": "serial_row", "cuda_optimized": "block_row"},
}


def load_cases(path):
    config = json.loads(Path(path).read_text())
    cases = config["cases"]
    if not cases:
        raise ValueError("Config must contain at least one case")
    for case in cases:
        name, shape = case["kernel"], case["shape"]
        rank = 2 if name == "softmax" else 1
        if name not in WORKLOADS or len(shape) != rank:
            raise ValueError(f"Invalid workload/rank: {case}")
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"Invalid dimensions: {shape}")
        if name == "softmax" and shape[-1] == 0:
            raise ValueError("Softmax cannot have zero columns")
    return cases


def prepare(workload, backend, inputs):
    """Return callable, completion fence, and conversion for correctness only."""
    import torch

    fn = implementation(workload, backend)
    device = inputs[0].device
    if backend == "jax":
        # Fail explicitly if only a CPU JAX runtime was installed for a GPU run.
        import jax

        from common.python.jax_bridge import from_torch, to_torch

        try:
            devices = jax.devices("gpu" if device.type == "cuda" else "cpu")
        except RuntimeError as exc:
            raise BackendUnavailable(f"JAX device unavailable: {device.type}") from exc
        if not devices:
            raise BackendUnavailable(f"JAX device unavailable: {device.type}")
        arrays = tuple(from_torch(x) for x in inputs)
        for array in arrays:
            array.block_until_ready()
        return lambda: fn(*arrays), lambda result: result.block_until_ready(), to_torch

    def synchronize(_):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    return lambda: fn(*inputs), synchronize, lambda result: result


def run_case(case, backend, args, inputs):
    import torch

    x = inputs[0]
    name = case["kernel"]
    record = {
        "kernel": name,
        "shape": case["shape"],
        "dtype": str(x.dtype).split(".")[-1],
        "implementation": backend,
        "variant": VARIANTS[name].get(backend, backend),
        "timing_mode": args.timing,
        "status": "pending",
    }
    try:
        if args.device == "cpu" and backend not in {"pytorch", "jax"}:
            raise BackendUnavailable("Backend requires NVIDIA CUDA")
        if args.timing == "cuda_event" and backend == "jax":
            raise BackendUnavailable("JAX uses a separate runtime/stream; use --timing wall")
        if x.dtype == torch.bfloat16 and x.is_cuda and not torch.cuda.is_bf16_supported():
            raise BackendUnavailable("Device does not support BF16")
        fn, synchronize, convert = prepare(name, backend, inputs)
        expected = implementation(name, "pytorch")(*inputs)
        actual = fn()  # Force compilation and compare before any measured samples.
        synchronize(actual)
        assert_correct(convert(actual), expected, inputs, name)
        measurement = measure(
            fn,
            synchronize,
            warmup=args.warmup,
            repetitions=args.repetitions,
            mode=args.timing,
            device=str(x.device),
        )
        model = work_model(name, case["shape"], x.element_size())
        record.update(
            status="ok",
            correctness="passed",
            **measurement,
            work_model=model,
            metrics=derived_metrics(
                model, measurement["latency_us"]["median"], args.peak_bandwidth
            ),
        )
    except BackendUnavailable as exc:
        record.update(status="skipped", reason=str(exc))
    except Exception as exc:
        # Compiler, correctness, OOM, and launch errors are failures, never skips.
        record.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    return record


def parser(default_kernel=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", choices=WORKLOADS, default=default_kernel)
    p.add_argument("--config", type=Path, default=CONFIG_DIR / "smoke.json")
    p.add_argument("--shape", type=int, nargs="+", help="One shape; requires --kernel")
    p.add_argument(
        "--implementations", nargs="+", choices=IMPLEMENTATIONS, default=list(IMPLEMENTATIONS)
    )
    p.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    p.add_argument("--device", default="cuda:0", help="cuda:N, or cpu for tooling validation")
    p.add_argument("--timing", choices=("wall", "cuda_event"), default="wall")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repetitions", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--peak-bandwidth", type=float, help="User-sourced theoretical GB/s (decimal)")
    p.add_argument("--bandwidth-source", help="Source URL/spec for the peak bandwidth value")
    p.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    p.add_argument("--require-all", action="store_true", help="Treat skipped backends as failure")
    return p


def main(default_kernel=None):
    p = parser(default_kernel)
    args = p.parse_args()
    if args.warmup < 1 or args.repetitions < 2:
        p.error("warmup must be >= 1 and repetitions >= 2")
    if args.peak_bandwidth is not None and (
        not math.isfinite(args.peak_bandwidth)
        or args.peak_bandwidth <= 0
        or not args.bandwidth_source
    ):
        p.error("Peak bandwidth must be positive and finite and needs --bandwidth-source")
    if args.shape is not None and not args.kernel:
        p.error("--shape requires --kernel")
    if args.output.exists():
        p.error(f"Refusing to overwrite {args.output}; choose a new output path")
    try:
        import torch
    except ImportError:
        p.error("Install PyTorch first: pip install -e '.[torch]'")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        p.error("Supported devices: cpu or cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            p.error("NVIDIA CUDA unavailable; use --device cpu for reference/tooling validation")
        torch.cuda.set_device(device)
    elif args.timing == "cuda_event":
        p.error("CUDA event timing requires a CUDA device")
    args.device = str(device)
    cases = load_cases(args.config)
    if args.kernel:
        cases = [c for c in cases if c["kernel"] == args.kernel]
    if args.shape is not None:
        if len(args.shape) != (2 if args.kernel == "softmax" else 1):
            p.error("Invalid shape rank")
        if any(n < 0 for n in args.shape) or (args.kernel == "softmax" and args.shape[-1] == 0):
            p.error("Invalid shape dimensions")
        cases = [{"kernel": args.kernel, "shape": args.shape}]
    if not cases:
        p.error("No cases selected")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    records = []
    # Import bridge before metadata capture so the effective allocator setting is recorded.
    if "jax" in args.implementations:
        from common.python import jax_bridge  # noqa: F401
    metadata = collect(args.device)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    payload = {
        "schema_version": 1,
        "environment": metadata,
        "settings": settings,
        "cache_policy": "reused inputs; no cache flush",
        "allocation_policy": "functional API; output/scratch allocations included",
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for case in cases:
            for dtype_name in args.dtypes:
                dtype = getattr(torch, DTYPES[dtype_name])
                inputs = tuple(
                    torch.randn(case["shape"], device=device, dtype=dtype)
                    for _ in range(2 if case["kernel"] == "vector_add" else 1)
                )
                order = list(args.implementations)
                rng.shuffle(order)
                for backend in order:
                    record = run_case(case, backend, args, inputs)
                    records.append(record)
                    print(
                        f"{case['kernel']} {case['shape']} {dtype_name} "
                        f"{backend}: {record['status']}"
                    )
                    # Preserve completed records even if a later run is interrupted.
                    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
                    temporary.replace(args.output)
    failed = any(r["status"] == "failed" for r in records)
    skipped = any(r["status"] == "skipped" for r in records)
    if failed or (args.require_all and skipped) or not any(r["status"] == "ok" for r in records):
        sys.exit(1)


if __name__ == "__main__":
    main()
