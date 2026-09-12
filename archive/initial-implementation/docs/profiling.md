# Profiling

Use Nsight Systems first to inspect the timeline, then Nsight Compute for a few
representative kernel launches. The profile target compiles, checks correctness,
warms up, synchronizes, and only then enters a CUDA profiler API capture region.
JAX profiling is outside this wrapper because it uses a separate runtime; use
JAX's official profiler tools for XLA traces.

```bash
bash scripts/profile.sh nsys --kernel vector_add --implementation cuda_optimized --shape 67108864
bash scripts/profile.sh ncu --kernel reduction --implementation cuda_naive --shape 16777219
bash scripts/profile.sh ncu --kernel reduction --implementation cuda_optimized --shape 16777219
```

Reports go to timestamped paths in `artifacts/profiles/`. The wrapper uses
`SpeedOfLight`, `LaunchStats`, `Occupancy`, `MemoryWorkloadAnalysis`, and
`SchedulerStats` sections, without requesting the full counter set. Section
availability can vary by Nsight release/GPU; inspect `ncu --list-sections` and
`ncu --query-metrics` on the target machine. GPU counter access may require a
machine administrator's configuration; do not change system permissions blindly.

## What to inspect

| Question | Evidence | Interpretation/caveat |
|---|---|---|
| Bandwidth limited? | DRAM throughput, bytes read/written, L2 throughput/hit rate | Reused inputs can be served from cache |
| Coalesced? | Global sectors/requests and memory transactions | Compare useful bytes with transaction bytes, including tails |
| Instruction limited? | SM throughput, issue activity, source/SASS load instructions | `float4` may reduce instruction count while bytes remain identical |
| Resource limited? | Registers/thread, shared memory/block, theoretical/achieved occupancy | More resident warps are useful only if they hide limiting latency |
| Synchronization limited? | Barrier stalls, eligible warps/cycle, scheduler issue activity | Compare the tree and shuffle reduction at identical launch geometry |
| Divergence/tails? | Active threads per executed warp, branch efficiency, source counters | Metric names vary; tiny rows naturally leave lanes idle |
| Shared-memory conflicts? | Shared bank conflicts, transactions per request | Separate replay/conflicts from necessary multi-warp traffic |
| Spilling? | Local-memory load/store traffic, registers/thread | Large padded Triton rows may spill or lose residency |
| Launch bound? | Nsight Systems CUDA API and kernel timeline, gaps, launch count | A two-pass reduction can lose on small N despite faster device work |

Start with the sections above. Add a focused `SourceCounters` section and inspect
source/SASS if instruction or stall attribution remains unclear. Sampling a
stall reason does not alone establish causality; change one factor and compare.

## Memory and synchronization validation

On a CUDA development host with Compute Sanitizer installed:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 python -m benchmarks.profile_workload \
  --kernel vector_add --implementation cuda_optimized --shape 1027 --iterations 1
compute-sanitizer --tool racecheck --error-exitcode 1 python -m benchmarks.profile_workload \
  --kernel reduction --implementation cuda_optimized --shape 4099 --iterations 1
compute-sanitizer --tool synccheck --error-exitcode 1 python -m benchmarks.profile_workload \
  --kernel softmax --implementation cuda_optimized --shape 17 129 --iterations 1
```

Run sanitizer checks separately from benchmarks. Profiling instrumentation,
replay, cache flushing, and synchronization perturb execution. Keep a normal
benchmark JSON alongside profiles rather than treating replay duration as latency.

## Evidence template

```text
Status: Results pending hardware benchmark
Source commit and exact command:
GPU / driver / Nsight versions:
Shape / dtype / implementation:
Expected bottleneck:
Measured benchmark change and dispersion:
Counter/section observations:
Alternative explanation:
Next falsifying experiment:
Relative path to raw profiler report:
```

Do not fill in a counter or claim until an actual capture supports it.

## Official references

- [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- [JAX profiling](https://docs.jax.dev/en/latest/profiling.html)
