# Local validation record

Validated during the initial implementation on macOS 26.3.1, ARM64, using
Python 3.14.7, PyTorch 2.14.0 (CPU), JAX/jaxlib 0.11.1 (CPU), NumPy 2.5.3,
Ruff 0.15.6, and clang-format 21.1.3. This is a CPU development environment,
not a supported NVIDIA performance measurement environment.

| Check | Observed result |
|---|---|
| CPU/reference/infrastructure pytest suite | 262 passed |
| GPU test collection on this machine | 648 cases skipped due to unavailable NVIDIA GPU(s) |
| Ruff lint and format | Passed |
| clang-format check on CUDA/C++ headers and sources | Passed |
| Shell syntax and pre-commit configuration | Passed |
| Editable package installation / dependency consistency | Passed |
| Source distribution and wheel build | Passed; CUDA sources/configs included |
| CMake with CUDA disabled | Configure/build passed; no CUDA compiled |
| PyTorch + JAX CPU benchmark smoke | 36 successful records: six shapes × three dtypes × two backends |
| Markdown and CSV generation from smoke JSON | Passed |
| Optional GPU backends on CPU benchmark | Explicit skips, no timing fields |
| CUDA-enabled CMake configuration | Cannot configure: `Failed to find nvcc` |

The CPU suite includes vector tails/alignment offsets, empty inputs, reduction
cancellation, softmax exceptional values, timing statistics, unit conversions,
report matching, repository structure, and correctness/build failure gating.
GPU stream/device/launch checks are present but have not been executed.

Local smoke JSON, reports, package-build log, environment inventory, and package
freeze live in ignored `artifacts/`. CPU timings are deliberately not committed
as GPU evidence. No benchmark or profiler numbers are published in this repository.

## Required NVIDIA validation

1. Build all three CUDA extensions with JIT and the optional CMake path.
2. Run `python -m pytest -m gpu --require-gpu-backends -q` on an SM80+ machine
   with compatible PyTorch, Triton, JAX, toolkit, and driver installations.
   The three multi-device cases additionally need two GPUs.
3. Run the memory/race/synchronization sanitizer commands in `profiling.md`.
4. Run the smoke/full shape sweeps and inspect thermal/load/cache conditions.
5. Capture focused Nsight Systems/Compute evidence before making speedup claims.

CUDA compilation, CUDA/Triton execution, JAX GPU interoperability, BF16 hardware
behavior, Nsight captures, and actual bandwidth/latency remain unvalidated.
GitHub-hosted CI is separate from these local checks; see the repository's Actions
tab for its current status. The manual GPU workflow needs a provisioned runner.

Results pending hardware benchmark
