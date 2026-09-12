#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tool="${1:-}"
if [[ "$tool" != ncu && "$tool" != nsys ]]; then
  echo "Usage: $0 {ncu|nsys} --kernel NAME --implementation NAME --shape DIMS..." >&2
  exit 2
fi
shift
command -v "$tool" >/dev/null || { echo "$tool is not installed" >&2; exit 1; }
mkdir -p artifacts/profiles
stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
if [[ "$tool" == ncu ]]; then
  exec ncu --target-processes all --profile-from-start off \
    --section SpeedOfLight --section LaunchStats --section Occupancy \
    --section MemoryWorkloadAnalysis --section SchedulerStats \
    --export "artifacts/profiles/ncu-$stamp" \
    "${PYTHON:-python}" -m benchmarks.profile_workload "$@"
else
  exec nsys profile --trace cuda,nvtx,osrt --sample none \
    --capture-range cudaProfilerApi --capture-range-end stop \
    --output "artifacts/profiles/nsys-$stamp" \
    "${PYTHON:-python}" -m benchmarks.profile_workload "$@"
fi
