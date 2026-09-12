# Future benchmark organization

Planning only. The active repository currently has no benchmark runner.

Add per-problem shape/dtype/layout cases under the problem's `benchmarks/` when
implementation begins. Extract a shared runner here only after its timing and
allocation contracts are clear. Future outputs should use JSON as the source
of truth, with raw samples, environment metadata, and explicit statuses.

Markdown/CSV/plots should be generated views of real results. Do not create
example performance tables with invented numbers. See the
[benchmarking plan](../docs/benchmarking.md) before designing the API.
