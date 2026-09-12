# GPU Performance Lab

GPU kernel implementation, profiling, and optimization experiments across CUDA,
Triton, PyTorch, and JAX.

This lab investigates how high-level ML operations map onto GPU hardware. Each
experiment starts with a correct baseline, identifies an expected bottleneck,
changes the execution strategy, and provides a reproducible path to test the
explanation with timing and profiler evidence.

**Status:** three forward workloads implemented. NVIDIA execution and performance
validation are pending. **Results pending hardware benchmark.** Implementation
coverage below describes code availability, not measured speedups or GPU validation.

| Kernel | PyTorch | JAX/XLA | Triton | CUDA | Optimization notes |
|---|:---:|:---:|:---:|:---:|---|
| [Vector add](kernels/01_vector_add) | ✅ | ✅ | ✅ | ✅ | Coalesced scalar → aligned FP32 `float4` |
| [Reduction](kernels/02_reduction) | ✅ | ✅ | ✅ | ✅ | Shared tree → warp shuffles; FP32 accumulation |
| [Softmax](kernels/03_softmax) | ✅ | ✅ | ✅ | ✅ | Serial row → cooperative row; stable reduction |
| LayerNorm / RMSNorm | planned | planned | planned | planned | Reduction, fusion, register reuse |
| GEMV / GEMM | planned | planned | planned | planned | Data reuse, tiling, Tensor Cores |
| Transformer primitives | planned | where useful | planned | planned | Fusion, layout, cache behavior |
| FlashAttention-like attention | planned | where useful | planned | planned | IO-aware tiled attention |

## Motivation

Performance engineering requires more than a fast kernel on one input. The useful
questions are where time goes, which hardware resource limits progress, how the
answer changes with shape and precision, and whether an optimization remains
correct at boundaries. The lab emphasizes memory traffic, synchronization,
instruction issue, occupancy, register pressure, and kernel launch overhead.

Read the [vector addition analysis](kernels/01_vector_add/README.md) for the
smallest complete experiment, then [reduction](kernels/02_reduction/README.md)
and [softmax](kernels/03_softmax/README.md) for progressively richer execution models.

## Repository structure

```text
common/          Timing, correctness, metadata, CUDA helpers, backend loading
kernels/         Three implemented experiments and eleven documented future workloads
benchmarks/      Shared runner, JSON configs, Markdown/CSV reporting, profiler target
docs/            Benchmarking, profiling, methodology, hardware, validation record
scripts/         Environment setup, benchmarking, Nsight entry points
tests/           CPU infrastructure, input contracts, CUDA stream checks
.github/         CPU CI and explicitly dispatched self-hosted GPU validation
```

Each implemented workload contains its own README, Python backends, benchmark
entry point, correctness tests, CUDA sources/bindings/CMake target, and results
directory. CUDA is loaded lazily through `torch.utils.cpp_extension`; an optional
CMake build produces the same importable extension modules.

## Setup

Use Python 3.11+ on an NVIDIA Linux host for all backends. Install a CUDA-enabled
PyTorch wheel appropriate to your driver, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[torch,dev]'
python -m pip install -e '.[triton,jax]'  # Optional comparisons; JAX extra targets CUDA 12
```

The first CUDA invocation builds the extension and requires `nvcc`, a compatible
host compiler, and Ninja. Builds and JIT compilation happen before measurement.
See [hardware/setup details](docs/hardware.md) for compatibility and CMake commands.

## Benchmark methodology

Every timed implementation must first pass correctness against the same
PyTorch inputs. Runs exclude compilation, include warmup, and retain raw latency
samples plus p20/p50/p80, mean, standard deviation, source revision, and environment
metadata. Inputs remain device-resident. Implementation order is shuffled using
a recorded seed; output and scratch allocation are part of the functional API.

Two timing scopes are available: synchronized host latency for comparisons across
all runtimes, and CUDA events for PyTorch/Triton/CUDA on the PyTorch stream. They
are reported separately. Effective GB/s uses minimum algorithmic traffic and
must not be interpreted as measured DRAM throughput. See
[benchmarking](docs/benchmarking.md) and [methodology](docs/methodology.md).

## Running benchmarks

```bash
python -m benchmarks.run_all --output artifacts/smoke.json
python -m benchmarks.run_all --config benchmarks/configs/full.json \
  --implementations pytorch triton cuda_naive cuda_optimized \
  --timing cuda_event --output artifacts/device.json
python -m kernels.01_vector_add.benchmark --shape 16777219 \
  --dtypes fp32 --output artifacts/vector-add.json
python -m benchmarks.report artifacts/smoke.json --output artifacts/smoke.md
python -m benchmarks.report artifacts/smoke.json --format csv --output artifacts/smoke.csv
```

Use a new output path for each run. Missing optional backends are recorded as
skips; correctness/build/runtime failures make the command fail. `--require-all`
also rejects skips. CPU reference/tooling smoke tests are available with
`--device cpu --implementations pytorch jax`; these are not GPU performance results.

## Running tests

```bash
python -m pytest -m 'not gpu' -q
python -m pytest -m gpu --require-gpu-backends -q  # Provisioned NVIDIA host
ruff check .
ruff format --check .
```

Tests cover FP32/FP16/BF16, scalar and empty vectors, tails, non-power-of-two
lengths, large tensors, storage offsets, cancellation, exceptional softmax
values, and nondefault CUDA streams. Reduction returns FP32; other operations
preserve input dtype. Operations are forward-only and require contiguous tensors.

## Profiling

```bash
bash scripts/profile.sh ncu --kernel reduction --implementation cuda_optimized --shape 16777219
bash scripts/profile.sh nsys --kernel softmax --implementation triton --shape 4096 4096
```

Capture starts after compilation, correctness, and warmup. The Nsight Compute
entry point selects focused throughput, launch, occupancy, memory, and scheduler
sections. [Profiling notes](docs/profiling.md) explain metrics, replay effects,
and how to attach evidence to an optimization claim.

## Supported hardware

The intended GPU target is Linux with NVIDIA CUDA, with SM80+ recommended for
native BF16 across backends. FP32/FP16 CUDA code uses warp shuffles and no
architecture-specific asynchronous copy instructions. Actual support also
depends on the installed PyTorch, Triton, JAX, toolkit, and driver versions.
No NVIDIA GPU has yet been validated for this repository. macOS/CPU can run
infrastructure and PyTorch/JAX reference tests; Apple MPS is outside this lab's scope.

## Kernel roadmap

| Phase | Workloads | Hardware questions |
|---|---|---|
| 1 — fundamentals | Vector add, reduction, scan, histogram | Bandwidth, barriers, warp communication, atomics |
| 2 — ML primitives | Softmax, LayerNorm, RMSNorm, GEMV, GEMM, fused activation | Reduction fusion, data reuse, Tensor Cores |
| 3 — transformers | RoPE, SwiGLU, cross entropy, quantization/dequantization, KV cache, attention | Layout, launch overhead, irregular memory, numerical stability |
| 4 — advanced | FlashAttention-like kernels, paged attention, quantized GEMM, persistent kernels, MoE routing, fused transformer operations | IO complexity, asynchronous copies, resource scheduling |

Future directories contain problem statements, performance hypotheses, and
acceptance criteria. They contain no placeholder implementations presented as working code.

## Hardware notes

Record GPU clocks, power limit, temperature, other GPU activity, and input working
set relative to L2 before interpreting a result. A memory-bound large vector can
be launch-bound at small sizes. Higher occupancy is a means of hiding latency,
not a target independent of register reuse and instruction count. Shared memory
is useful when it reduces communication costs or global traffic, not merely
because it is on chip.

## Reproducibility

JSON is the measurement source of truth; Markdown and CSV are generated views.
Each run records library versions, driver inventory, CUDA build version, GPU
properties, Python/OS, seed, timing settings, and Git revision/dirty state.
Save `python -m pip freeze` alongside publishable runs. Dependency ranges express
compatibility intent, not a tested cross-product; attach an exact environment to
each measured result. See [local validation](docs/validation.md) for what has
actually been exercised and [CONTRIBUTING.md](CONTRIBUTING.md) for evidence requirements.

MIT licensed. The project is independent and has no affiliation with a GPU vendor
or AI laboratory.
