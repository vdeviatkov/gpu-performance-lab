# 01 · Vector Addition

A first streaming-kernel study: distinguish launch costs from memory throughput,
then test whether vectorized access reduces instruction cost.

## Source and contract

[LeetGPU Vector Addition](https://leetgpu.com/challenges/vector-addition), Easy.
Inspected on 2026-09-12: equal-length FP32 vectors, `C[i] = A[i] + B[i]`,
1 ≤ N ≤ 100,000,000; the performance case uses N = 25,000,000.
These are library-integrated experiments, not challenge submissions.

**Lab extensions:** empty vectors, FP16/BF16, and contiguous storage-offset views.
The checked entry point is `api.add(a, b, backend=..., out=None)`:

- Equal contiguous 1-D tensors, matching dtype/device; no broadcasting.
- FP32 arithmetic, with FP16/BF16 outputs rounded after an FP32 intermediate.
- Inputs may alias each other; output must not overlap either input. Disjoint
  slices of the same storage are allowed. Inputs are never mutated.
- Fresh output unless `out` is supplied; empty inputs launch no device kernel.
- Forward-only: autograd and unresolved negative/conjugate views are rejected.
- CUDA/Triton use the input device and PyTorch's current stream, asynchronously.
  Consumers on other streams must establish their own dependencies.
- CPU PyTorch/JAX support correctness checks, not GPU performance claims.

No prerequisites. Primary lesson: **launch overhead vs bandwidth**.

## Implementations

| Backend | Mechanism |
|---|---|
| `pytorch` | Native eager add, with optional output buffer |
| `cuda_scalar` | One element per thread, coalesced across lanes |
| `cuda_grid_stride` | Coalesced loop, capped at 4096 blocks |
| `cuda_float4` | Four aligned FP32 elements per thread, scalar tail |
| `triton` | Masked tiles; default 1024 elements and 4 warps |
| JAX | Native arrays with JIT compilation and explicit completion |

Read [CUDA kernels](cuda/kernels.cu), [bindings](cuda/bindings.cpp),
[Triton](triton/implementation.py), [PyTorch](pytorch/implementation.py),
and [JAX](jax/implementation.py). All support FP32/FP16/BF16.
The float4 variant falls back to grid-stride for low-precision or misaligned buffers.

[api.py](api.py) owns the public PyTorch-tensor contract; backend modules are
internal. JAX exposes a separate native-array interface. CUDA repeats validation
at the C++ boundary, uses an RAII device guard and checks launch errors without
inserting synchronization.

## Performance model

N additions move a minimum of `3*N*s` logical bytes for element size s: two
reads and one write. Arithmetic intensity is `1/(3*s)` FLOP/byte, or 1/12
for FP32. At N = 25,000,000, logical traffic is 300 MB. These are algorithmic
estimates, not measurements.

At a bandwidth ceiling W bytes/s, `3*N*s/W` is an optimistic memory-time bound.
Large working sets should stress memory throughput; tiny vectors can be dominated
by launches and dispatch. Cached inputs may stress L2 rather than DRAM.
Effective GB/s is `3*N*s / seconds / 1e9`, not measured DRAM traffic.
There is no data reuse to justify shared memory and no Tensor Core operation here.

## Optimization journey

1. **Scalar coalesced:** establish the launch/bandwidth curve.
2. **Grid stride:** reuse threads with a capped grid. Less scheduling work may
   help, but loops add instructions. The 4096-block cap is an explicit starting
   choice, not a hardware-derived optimum. Sweep 128/256/512 threads.
3. **Aligned float4:** reduce memory/address instruction count while moving the
   same bytes. All three pointers must be 16-byte aligned. At most three trailing
   floats are handled once in the same launch. Inspect SASS to verify vector
   instructions; extra registers may erase instruction savings.
4. **Triton:** sweep 256/512/1024/2048 elements and 4/8 warps. Explicit masks
   protect tails; no hidden autotuning enters measurement.

Compare latency distributions, executed instructions, DRAM/L2 throughput,
registers and eligible warps. A scalar kernel may already saturate bandwidth.
No variant is claimed faster until measured.

## Setup and usage

From the repository root, use Python 3.11+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r problems/01_vector_add/requirements.txt
```

On Linux NVIDIA hosts, use a CUDA-enabled PyTorch wheel, matching driver/toolkit
and supported C++ compiler. Make nvcc discoverable through CUDA_HOME or PATH.
Install Triton separately if PyTorch does not supply it. JAX is optional; install
the CUDA extra appropriate to your setup. See the
[PyTorch selector](https://pytorch.org/get-started/locally/) and
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

The [loader](cuda/implementation.py) builds lazily with Ninja via
[PyTorch's extension API](https://docs.pytorch.org/docs/stable/cpp_extension.html).
It inherits PyTorch's required C++ standard and ABI, uses O3/line information,
and does not enable fast math. Set `TORCH_EXTENSIONS_DIR=artifacts/torch_extensions`
to keep the build cache here. Record any explicit `TORCH_CUDA_ARCH_LIST`.

```python
from importlib import import_module
import torch

add = import_module("problems.01_vector_add.api").add
a = torch.randn(1025, device="cuda")
b = torch.randn_like(a)
c = add(a, b, backend="cuda_float4")
torch.cuda.synchronize()
```

## Correctness tests

```bash
python -m pytest -q
ruff check problems/01_vector_add
ruff format --check problems/01_vector_add
clang-format --dry-run --Werror problems/01_vector_add/cuda/*.{cpp,cu}
```

Tests cover empty/odd/large vectors, tails, independent output alignment,
guard values, input preservation, cancellation and NaN/Inf, invalid contracts,
launch configurations, nondefault streams and multiple devices.
The reference adds FP32-converted inputs before rounding to the output dtype.
Tolerances are explicit in api.py; guard/ownership checks are exact.
Missing hardware/dependencies are explicit skips; build and correctness failures
fail. Inspect skips on the target host before acceptance.

## Benchmarking

```bash
python -m problems.01_vector_add.benchmarks.run --output artifacts/vector_add.json
python -m problems.01_vector_add.benchmarks.report artifacts/vector_add.json

# Include JAX with a common host-observed completion scope.
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m problems.01_vector_add.benchmarks.run \
  --scope api --backends pytorch cuda_scalar cuda_grid_stride cuda_float4 triton jax \
  --output artifacts/vector_add_api.json

# Misalignment and explicit launch geometry.
python -m problems.01_vector_add.benchmarks.run --offset 1 --threads 128 \
  --block-size 512 --num-warps 4 --output artifacts/vector_add_offset.json
```

Each backend passes correctness before warmup/timing. Default sizes are 1, 1025,
1,048,576 and 25,000,000; dtypes are FP32/FP16/BF16. BF16 benchmarks require SM80+
by policy. Requested missing packages and build/runtime errors abort the run;
select available implementations with `--backends`.

- **device:** preallocated output, CUDA events on the current stream, synchronized
  after each sample. This measures stream elapsed time around an eager call;
  host submission gaps can enter it. It is not isolated kernel duration.
- **api:** host-observed allocation, dispatch and completion. PyTorch/CUDA/Triton
  synchronize the selected device; JAX waits on its result. Transfers and JIT/build
  are excluded. Tiny comparisons include runtime-specific completion costs.
  JAX receives the exact GPU inputs through DLPack, never through a CPU fallback.

Never mix these scopes in a speedup claim. Defaults are 25 warmups and 100 samples,
with raw microseconds, p20/p50/p80, mean/stddev, effective GB/s, elements/s and
matching-case speedup against PyTorch in JSON. Backend order is seeded/shuffled
per case. Repeat whole runs to assess noise and ordering effects.

Buffers are reused without flushing. Compare footprint to recorded L2 size and
sweep larger sizes before claiming DRAM saturation. Rotating buffers, CUDA Graphs
and sourced theoretical-bandwidth percentages remain future experiments.
Metadata includes GPU, driver/toolkit/runtime, installed packages, source revision
and dirty state, configuration and `--notes` for clocks/load/thermal conditions.
Unavailable metadata is null; no hardware ceiling is guessed.

## Profiling

Build/warm up before a CUDA profiler capture range. Choose one backend, size and
dtype; profile runs produce no benchmark JSON.

```bash
mkdir -p artifacts
ncu --profile-from-start off --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --section LaunchStats --launch-count 1 -o artifacts/vector_add_ncu \
  python -m problems.01_vector_add.benchmarks.run --profile --backends cuda_float4 \
  --sizes 25000000 --dtypes fp32 --samples 1

nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi \
  --capture-range-end=stop -o artifacts/vector_add_nsys \
  python -m problems.01_vector_add.benchmarks.run --profile --backends cuda_scalar \
  --sizes 1025 --dtypes fp32 --samples 100

compute-sanitizer --tool memcheck python -m pytest -q \
  problems/01_vector_add/tests/test_vector_add.py -k 'values_and_guards and cuda'
```

Nsight Systems separates launches from device execution; Nsight Compute tests
bandwidth/instruction hypotheses. Compare scalar/float4 SASS and resources;
add InstructionStats or SchedulerStats only as needed. Replay/cache policies
can alter observations: retain profiler configuration and time normal runs separately.

## Results and status

🟨 **In Progress — implemented; NVIDIA validation pending.**

Results pending hardware benchmark.

Local checks use macOS/Apple Silicon, Python 3.14 and PyTorch 2.14, with CPU JAX
available. Local validation: **132 passed, 486 hardware-dependent skips**;
Python lint/formatting and C++ formatting checks pass. CUDA device code,
Triton execution, GPU JAX, sanitizer/profiler runs and performance are unvalidated:
this host has no NVIDIA GPU or CUDA toolkit. The host binding passed a C++20
syntax check against local PyTorch headers; it is not a CUDA build.

Next: run tests and sanitizer on NVIDIA hardware, capture the first benchmark
matrix, then investigate float4 crossover sizes and cache sensitivity.

## References

- [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Triton programming model](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [JAX asynchronous benchmarking](https://docs.jax.dev/en/latest/benchmarking.html)
- [Lab methodology](../../docs/methodology.md) and [benchmark policy](../../docs/benchmarking.md)
