# Preserved initial implementation

The complete earlier repository snapshot is preserved unchanged under
[`initial-implementation/`](initial-implementation/README.md), from commit
`dd2d199` (the initial CUDA/Triton/PyTorch/JAX implementation pass).
All 96 previously tracked files were verified byte-for-byte after the move.

That earlier scope was superseded by the curriculum-only request. Its kernels,
benchmark code, tests, build configuration, workflows, and documentation are
historical prototypes, not current curriculum solutions or accepted evidence.
No new kernel or benchmark implementation was added during this reorganization.
The archived `.github/workflows` directory is intentionally outside GitHub's
active workflow location.

The new `problems/` roadmap starts at **Planned**. Vector addition, reduction,
and softmax have archived prototypes, but they have not been accepted against
the new source-specific contracts or validated on NVIDIA hardware. Revisit them
as reference material when beginning those problems; do not promote their status
automatically. Run archive tooling only from its directory with a separately
understood environment. Its old README describes its historical scope.

Existing ignored local environments, build products, and smoke artifacts remain
on disk. They are not part of the planning scaffold or new benchmark evidence.
