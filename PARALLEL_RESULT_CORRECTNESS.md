# Kokkos parallel execution — verified numerical result divergence

Date: 2026-11-03 JST
Research branch: `research/core-arithmetic-audit-20261103`
Pinned Kokkos commit: `1e08552799ab366f3037ce7ad46cc8d3ca85498d`
Initial workflow: https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41617987251
Focused probe: `parallel_result_probe.cpp`
All builds and execution on GitHub Actions Linux runners only; upstream PR #9725 not touched.

## Test objective

Test the value ultimately observed by a scientific application: number of actual functor calls, the set of indices reached, whether any index falls outside the declared domain, and numerical sum produced by a real `parallel_reduce`. Use host `OpenMP` and `Threads` on distinct backends, `Kokkos::initialize(...set_num_threads(4))`, real Kokkos execution, atomically updated observation counters, no out-of-range memory writes, and no giant workload or allocations.

## Verified results: optimized actual production API calls

| Backend | Compiler | Workload | Expected | Observed | Result |
| --- | --- | --- | --- | --- | --- |
| OpenMP | GCC | ordinary `RangePolicy parallel_for` 1024 elements | 1024 calls, exactly once/index | 1024 calls, 0 OOB, 0 missing/duplicate | PASS |
| OpenMP | GCC | `RangePolicy parallel_for` with 1 logical element `[INT32_MAX-1,INT32_MAX)` | 1 call | 1 call, 0 OOB | PASS |
| OpenMP | GCC | `MDRangePolicy parallel_for`, 1 logical element, explicit tile `{4,1}` | 1 call | **4 calls, 3 out-of-bounds** | WRONG |
| OpenMP | GCC | `MDRangePolicy parallel_reduce`, same range | reduction value 1 | **reduction 4**, 4 calls, 3 OOB | WRONG |
| Threads | GCC | ordinary `RangePolicy parallel_for` 1024 elements | 1024 calls, once/index | 1024 calls, 0 OOB, 0 missing/duplicate | PASS |
| Threads | Clang | ordinary `RangePolicy parallel_for` 1024 elements | 1024 calls, once/index | 1024 calls, 0 OOB, 0 missing/duplicate | PASS |
| Threads | GCC | `RangePolicy parallel_for` 1 logical element `[INT32_MAX-1,INT32_MAX)` | 1 call | **3 calls, 2 out-of-bounds** | WRONG |
| Threads | Clang | same | 1 call | **3 calls, 2 out-of-bounds** | WRONG |
| Threads | GCC, Clang | `MDRangePolicy parallel_for` 1 logical element | 1 call | **4 calls, 3 out-of-bounds** | WRONG |
| Threads | GCC, Clang | `MDRangePolicy parallel_reduce` 1 logical element | reduction 1 | **reduction 4**, 4 calls, 3 OOB | WRONG |

## Sanitizer evidence

- OpenMP + GCC, both MDRange variants: `KokkosExp_Host_IterateTile.hpp:1221`, `2147483646 + 4` signed overflow.
- Threads + GCC and Clang, RangePolicy 1-element case: `Kokkos_ExecPolicy.hpp:338`, `2147483646 + 3` signed overflow.
- Threads + GCC and Clang, both MDRange cases: same HostIterateTile overflow.

The OpenMP/Clang job in the initial workflow failed at CMake configuration because the default runner lacked Clang's OpenMP runtime. This is a toolchain limitation, **not Kokkos correctness evidence**. Follow-up CI matrix excludes this unsupported combo. Diagnostic cases are allowed to return nonzero so that later cases run; do not interpret workflow-level success as correctness.

## What is and is not proven

**Proven:** For the tested Kokkos version and boundary inputs, real multithreaded host production APIs can execute outside the declared iteration domain and return an incorrect reduction value. This is a real numerical-correctness failure, not only an arithmetic sanitization warning.

**Not proven:** An unconstrained inter-thread data race, overlapping writes to the same application memory address, GPU failure, typical-range failure, historic scientific results being corrupted, or an exploitable memory-safety issue. The test observes indices and counters atomically and intentionally does not perform out-of-bounds memory writes.

**Possible consequence:** Real applications whose functors index memory by their Kokkos-provided coordinates might perform out-of-domain accesses if they do not independently guard the index, but this remains an impact hypothesis, not a demonstrated memory-corruption result.

## Next evidence gates

1. Add `RangePolicy parallel_reduce` 1024-element baseline and 1-element high-index reduction; validate whether Threads returns a wrong numerical sum for this separate defect.
2. Repeat selected cases to distinguish deterministic arithmetic partition failures from scheduling-dependent nondeterministic overlap.
3. Perform isolated A/B patches on pinned source in the research repo, verifying that original baseline behavior is preserved and boundary calls/count/reduction become correct.
4. Review whether internal OpenMP HostThreadTeam work partition exhibits an independent large-index issue; it is not the cause of the one-element OpenMP RangePolicy control.
5. Do not broaden upstream #9725 or claim all historical computations are affected without corresponding input conditions and backend confirmation.

## Follow-up: RangePolicy actual numerical reduction is also wrong on Threads

Additional run: https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41618217187

With `RangePolicy<Threads, IndexType<int32_t>>`, `[INT32_MAX-1, INT32_MAX)`, explicit `ChunkSize(1)`, a real `Kokkos::parallel_reduce` returning the number of visits reports **3 instead of the correct 1**. This occurs under GCC and Clang optimized builds with 4 host threads. Exactly 3 functor calls occurred; 2 indices were out of the declared range. UBSan in both compilers points at `Kokkos_ExecPolicy.hpp:338`. Ordinary 1024-element reduction correctly reports 1024. No data races were needed to produce this deviation: the test uses atomic counters and the documented reduction update.

Therefore **both independent problems** (MDRangePolicy HostIterateTile and RangePolicy WorkRange on Threads) have confirmed wrong numerical reduction results on real parallel execution paths. This is distinct from the hypotheses about overlapping application writes, which remain untested.

## Additional observation: OpenMP high-origin RangePolicy reduction timeout

In completed run https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41618241871 , OpenMP/GCC `boundary-range-reduce` did not finish within the configured **20-second timeout** in either the optimized or UBSan build. The equivalent OpenMP `parallel_for` with 1 logical element finished with one correct call, and 1024-element reduction completed normally. The code used a public `RangePolicy<OpenMP, IndexType<int32_t>>` with bounds `[INT32_MAX-1, INT32_MAX)` and `ChunkSize(1)`. This observation is **not yet classified as a hang caused by overflow**; investigate with a smaller isolation matrix and inspect the OpenMP reduction dispatch, scheduler and whether the body is iterating an unexpectedly huge range. The test-runner timeout prevents indefinite execution.

The complete follow-up job matrix (OpenMP GCC; Threads GCC and Clang) compiled and executed. Workflow success means the harness ran and captured diagnostics; individual failures/timeouts are intentional evidence, not passing correctness tests.
