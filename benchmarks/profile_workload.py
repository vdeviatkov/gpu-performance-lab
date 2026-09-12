"""Compile and warm up before the CUDA profiler API capture region."""

import argparse

from benchmarks.run_all import DTYPES, prepare
from common.correctness.checks import assert_correct
from common.python.registry import WORKLOADS, implementation


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", choices=WORKLOADS, required=True)
    p.add_argument(
        "--implementation",
        choices=("pytorch", "triton", "cuda_naive", "cuda_optimized"),
        required=True,
    )
    p.add_argument("--shape", nargs="+", type=int, required=True)
    p.add_argument("--dtype", choices=DTYPES, default="fp32")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.iterations < 1:
        p.error("iterations must be positive")
    if not torch.cuda.is_available():
        p.error("An NVIDIA CUDA GPU is required")
    torch.cuda.set_device(args.device)
    torch.manual_seed(2026)
    with torch.inference_mode():
        inputs = tuple(
            torch.randn(args.shape, device=args.device, dtype=getattr(torch, DTYPES[args.dtype]))
            for _ in range(2 if args.kernel == "vector_add" else 1)
        )
        fn, sync, convert = prepare(args.kernel, args.implementation, inputs)
        result = fn()
        sync(result)
        assert_correct(
            convert(result), implementation(args.kernel, "pytorch")(*inputs), inputs, args.kernel
        )
        for _ in range(25):
            result = fn()
        sync(result)
        torch.cuda.profiler.start()
        try:
            with torch.cuda.nvtx.range(f"{args.kernel}/{args.implementation}"):
                for _ in range(args.iterations):
                    result = fn()
                sync(result)
        finally:
            torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
