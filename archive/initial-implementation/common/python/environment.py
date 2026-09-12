"""Capture versions without initializing optional runtimes."""

import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect(device=None):
    root = Path(__file__).resolve().parents[2]
    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "os": platform.platform(),
        "architecture": platform.machine(),
        "versions": {p: version(p) for p in ("torch", "triton", "jax", "jaxlib", "numpy")},
        "git_commit": command(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "git_dirty": None,
        "nvcc": command(["nvcc", "--version"]),
        "nvidia_smi": command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total,pstate,"
                "clocks.sm,clocks.mem,power.limit,temperature.gpu",
                "--format=csv,noheader",
            ]
        ),
        "environment": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "TORCH_CUDA_ARCH_LIST",
                "CUDA_MODULE_LOADING",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
            )
        },
        "hardware": {"gpu": None, "cuda": None, "device": device},
    }
    status = command(["git", "-C", str(root), "status", "--porcelain"])
    metadata["git_dirty"] = None if status is None else bool(status)
    inventory_fields = (
        "index",
        "name",
        "uuid",
        "driver",
        "memory_total",
        "pstate",
        "sm_clock",
        "memory_clock",
        "power_limit",
        "temperature",
    )
    metadata["gpu_inventory"] = [
        dict(zip(inventory_fields, (value.strip() for value in row), strict=True))
        for row in csv.reader((metadata["nvidia_smi"] or "").splitlines())
        if len(row) == len(inventory_fields)
    ]
    metadata["hardware"]["driver"] = (
        metadata["gpu_inventory"][0]["driver"] if metadata["gpu_inventory"] else None
    )
    if device is not None and str(device).startswith("cuda"):
        import torch

        props = torch.cuda.get_device_properties(device)
        metadata["hardware"].update(
            {
                "gpu": props.name,
                "cuda": torch.version.cuda,
                "compute_capability": [props.major, props.minor],
                "total_memory_bytes": props.total_memory,
                "multiprocessor_count": props.multi_processor_count,
                "uuid": str(getattr(props, "uuid", "")) or None,
            }
        )
    return metadata


def main():
    print(json.dumps(collect(), indent=2))


if __name__ == "__main__":
    main()
