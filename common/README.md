# Future shared utilities

Planning boundary only; no active utility framework exists yet.

When multiple studies need the same behavior, this directory may hold small
helpers for input contracts, dtype-specific correctness tolerances, CUDA error
handling, environment records, and timing. Keep allocations, launches, stream
selection, and synchronization visible rather than hiding them in a large layer.

Begin locally in the first study, then extract only demonstrated shared needs.
The [benchmarking](../docs/benchmarking.md) and [methodology](../docs/methodology.md)
documents define the future behavior. Earlier code is preserved in the archive.
