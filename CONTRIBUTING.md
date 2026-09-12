# Contributing a study

This repository is currently a curriculum scaffold. Do not add solution stubs,
empty backend files, generated timings, or infrastructure unrelated to the study
being started. Keep the roadmap, skill map, flagship list, and progress counts
consistent through simple manual edits.

## Start a problem

Read its source, prerequisites, primary lesson, and proposed experiments. Write
down the exact interface and numerical contract. Change status to 🟨 In Progress
in the problem README and root roadmap. Create only the backend directories
being used, with real implementations when that work is explicitly undertaken.

Use meaningful filenames for mechanisms, preserve baselines, and explain why
each backend comparison is useful. Python should use clear modules and pytest
tests; CUDA should use modern C++, explicit ownership, current-stream launches,
device guards where needed, and error checks. Introduce shared utilities only
when two studies need the same well-understood behavior.

## Complete a problem

Require the planned backends, correctness across the documented contract,
meaningful baseline comparisons, real GPU benchmark samples, environment and
commands, and an analysis of limitations. If the backend plan changes, explain
why and adjust the coverage denominators. Update the problem README, root table,
and [progress](docs/progress.md) together.

Use ✅ Complete for accepted study completion. Use 🚀 Optimized only when a named
change has measured performance evidence, appropriate profiler evidence, and an
explanation of both wins and regressions. Optimized counts as Complete once,
not as an additional problem. P0 work follows the deeper
[flagship requirements](docs/flagship.md).

## Preserve attribution and scope

Link LeetGPU statements rather than copying them or their solutions. Record
official title/difficulty separately from local aliases and difficulty. Label
custom problems **Portfolio Extension**. Source-conforming runs and broader
dtype/layout experiments must remain distinguishable.

Archived prototypes are available for later review, but they are not accepted
solutions. Do not modify the preserved snapshot as part of new curriculum work;
bring forward only the parts justified by a newly agreed study contract.
