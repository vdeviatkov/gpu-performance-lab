# Contributing

Keep each experiment small enough to audit. A change should connect a concrete
hardware hypothesis to a correctness-preserving implementation and a reproducible
measurement. Unmeasured work is welcome if its validation status is explicit.

## Development

```bash
python -m pip install -e '.[torch,dev]'
pre-commit install
ruff check . --fix
ruff format .
find common kernels -type f \( -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' \) -print0 | xargs -0 clang-format -i
python -m pytest -m 'not gpu' -q
```

Use C++17, current PyTorch streams, device guards, 64-bit indexing, explicit input
contracts, and launch error checks. Avoid global device synchronization inside
operation wrappers. Keep output precision and approximation choices visible.

## Adding an optimization

Preserve the baseline. Give the kernel a name describing its mechanism. Record
expected bottleneck, code change, expected hardware effect, benchmark status,
and profiler evidence in the workload README. If GPU hardware is unavailable,
write `Results pending hardware benchmark` and explain which checks remain.

Add boundary and adversarial cases that could break the optimization: vector
alignment/tails, inactive lanes, partial blocks, cancellation, infinities,
nondefault streams, or multi-device placement as relevant. Run the complete
GPU suite and sanitizer cases on the target hardware before claiming validation.
Do not swallow compilation or correctness failures as optional-backend skips.

## Adding a workload

Promote a documented future directory to the common implemented-workload layout.
Define semantics first; add registry/config entries, the PyTorch reference,
applicable backends, correctness tests, and a performance model. Update structure
validation to match the number of implemented directories if the roadmap grows.
Do not force JAX or another backend where it adds no useful comparison.

## Publishing results

Commit measured JSON, generated report, exact package freeze, commands, and
hardware notes together under the workload's `results/` directory. Profile
reports may be too large for Git; link a release artifact and record its checksum.
Use a clean source commit as the measurement revision and a later commit to add
results. Include slowdowns and noise, and disclose changed clocks or settings.
Never add example timings to committed result files. Small synthetic unit-test
fixtures are labeled as such and are not benchmark evidence.

CPU CI runs lint, formatting, reference/infrastructure tests, package builds, and
the CUDA-disabled CMake configuration. The manual GPU workflow requires a
maintainer-controlled self-hosted runner; it never executes untrusted fork PRs.
