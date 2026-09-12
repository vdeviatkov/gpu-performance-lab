# Future correctness organization

Planning only; no new test implementation is included.

Problem-specific tests should eventually live under each problem's `tests/`.
Shared tests here should cover only reusable contracts or infrastructure once
they exist. Use pytest for Python interfaces and appropriate GPU memory/race
checks for low-level changes.

Test valid boundaries, odd and non-power-of-two shapes, strides/alignment where
supported, precision, numerical extremes, ownership, and current-stream behavior.
Add input distributions for contention/sparsity/routing rather than only random
normal data. Record missing hardware as a skip, never a correctness success.

Configure test discovery to exclude `archive/` when the active harness is added.
The archived test suite is historical and must not be discovered as current work.
