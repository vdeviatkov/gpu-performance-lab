# Hardware and environment

## Compatibility intent

The complete lab targets NVIDIA GPUs on Linux, Python 3.11+, PyTorch 2.6+,
Triton 3.2+, JAX 0.5+, C++17, and CMake 3.24+. These are package ranges, not a
claim that every combination has been validated. Start with an environment
supported by the chosen PyTorch release, then match its CUDA wheel to a toolkit
and driver using the official compatibility documentation.

The `jax` package extra selects CUDA 12 on Linux. For CUDA 13 or another JAX
installation layout, install JAX separately following its current installation
guide. Do not install a CPU-only JAX runtime and label it a GPU comparison.
JAX GPU inputs use DLPack; conversion and completion are outside measured calls.
The lab defaults JAX preallocation off before JAX import to reduce competition
with PyTorch. Existing environment settings take precedence and are recorded.

CUDA extensions require `nvcc` in addition to PyTorch's runtime libraries.
An NVIDIA driver alone does not provide the compiler. CUDA code has no hardcoded
SM architecture; PyTorch JIT normally selects the visible GPU. Set
`TORCH_CUDA_ARCH_LIST` explicitly for portable build artifacts. Native BF16
across backends is best targeted at SM80+; unsupported BF16 tests are skipped.

## Build choices

JIT is the primary path: install the package and invoke a CUDA backend. The
extension cache is managed by PyTorch; `-O3 -lineinfo` is used without fast math.
All launches use a RAII device guard, current PyTorch stream, and immediate
launch error checks. Asynchronous execution errors surface at test/timing fences.
PyTorch tensors own output/scratch allocations; no raw host-side CUDA allocation
framework is needed.

For an explicit CMake build on a GPU host (replace `80` with your target SM):

```bash
cmake -S . -B build-cuda -DGPU_LAB_BUILD_CUDA=ON \
  -DPython3_EXECUTABLE="$(command -v python)" -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-cuda --parallel 2
PYTHONPATH="$PWD/build-cuda/python${PYTHONPATH:+:$PYTHONPATH}" \
  python -m pytest -m gpu --require-gpu-backends -q
```

The loader imports CMake-built `gpu_lab_*` modules first, otherwise JIT builds
them. A CPU-only configuration validates the build graph without compiling CUDA:

```bash
cmake -S . -B build-cpu
cmake --build build-cpu
```

## Capture hardware

```bash
python -m common.python.environment > artifacts/environment.json
nvidia-smi
nvcc --version
python -m pip freeze > artifacts/requirements-measured.txt
```

Create `artifacts/` first if running the standalone commands. Benchmarks embed
versions and GPU information automatically. `nvidia_smi` stores the full visible
inventory with index, UUID, driver, memory, P-state, clocks, power limit, and
temperature. Selected-device name, compute capability, memory, SM count, and
PyTorch CUDA build version are structured separately. Device indices can be
remapped by `CUDA_VISIBLE_DEVICES`; retain the inventory and environment field.

## Measurement hardware record

Results pending hardware benchmark

| Field | Value |
|---|---|
| Measured GPU / UUID | pending |
| SM architecture / L2 size | pending |
| Driver / toolkit / PyTorch CUDA build | pending |
| Nominal bandwidth and source | pending |
| SM/memory clocks / power policy | pending |
| Thermal state / concurrent activity | pending |
| OS / container / package freeze | pending |

## Official references

- [PyTorch installation selector](https://pytorch.org/get-started/locally/)
- [PyTorch C++ extensions](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [JAX installation](https://docs.jax.dev/en/latest/installation.html)
- [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [Triton installation](https://triton-lang.org/main/getting-started/installation.html)
