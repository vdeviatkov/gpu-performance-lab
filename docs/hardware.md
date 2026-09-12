# Hardware planning

No hardware performance claim or supported-device matrix has been established
for the active curriculum. Choose the GPU/toolchain when implementing a study,
then record the exact combination rather than relying on a generic CUDA label.

## Capabilities to record

- GPU model/UUID, compute capability, SM count, memory capacity, and L2 size.
- Documented peak bandwidth and compute ceilings for the precise arithmetic mode.
- Driver, CUDA toolkit/runtime, host compiler, Python, PyTorch, Triton, and JAX versions.
- FP16/BF16 and integer matrix capabilities, supported Tensor Core modes, and
  asynchronous-copy features for the target architecture.
- Power limit, clocks, temperature, other GPU activity, and execution environment.

Mixed precision is a numerical contract. Record input, accumulator, and output
dtypes separately. Tensor Core availability does not guarantee a kernel uses
them efficiently; layout, tile dimensions, instruction selection, and occupancy
must be measured. BF16, FP8, and asynchronous-copy experiments are conditional
on the actual hardware/software combination.

The curriculum should eventually include at least one bandwidth-oriented study
and one matrix-compute study on the same device. Multiple GPUs are valuable for
portability comparisons, but a single carefully documented device is sufficient
to begin. CPU reference results do not count as GPU performance evidence.

Use Python 3.11+ and modern C++17 or later when implementation begins. Prefer
small PyTorch extension/CMake integration only once needed. Do not install a
large dependency stack, add Docker, or create a build graph for empty directories.

The [archived validation record](../archive/initial-implementation/docs/validation.md)
describes the earlier Mac/CPU prototype checks. It is historical information,
not validation of these 50 current study contracts.
