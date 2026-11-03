# Numerical correctness and multithreaded execution audit (2026-11-03)

Source pinned to Kokkos commit `1e08552799ab366f3037ce7ad46cc8d3ca85498d`.
Workflow: https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41619708733
Probe: `numerical_concurrency_probe.cpp`.
Branch: `research/core-arithmetic-audit-20261103`.

**Objective:** Measure actual numerical divergence, not just UBSan. Use the real public `Kokkos::parallel_reduce` paths on OpenMP and Threads backends, run with 1, 2, 4 or 8 configured CPU threads. The application records visited signed index values and actual `std::thread::id` under a mutex, then compares to the declared domain. It never indexes memory at invalid coordinates. It sums `1.25` for every functor invocation, including invalid ones, to demonstrate the effect on reduction outputs. It is **not** an experiment performing out-of-bounds writes or proving a data race.

## Key results (optimized, real parallel reduction)

The boundary workload is **ONE** legal work element: index range `[INT32_MAX - 1, INT32_MAX)`. Correct numerical result is `1.25`.

| Backend | Pattern | Threads | Expected | Actual | Relative error | Out-of-domain calls | Observed distinct worker threads |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Threads (GCC and Clang) | RangePolicy | 1 | 1.25 | 1.25 | 0% | 0 | 1 |
| Threads (GCC and Clang) | RangePolicy | 2 | 1.25 | 1.25 | 0% | 0 | 1 |
| Threads (GCC and Clang) | RangePolicy | 4 | 1.25 | **3.75** | **+200%** | **2** | **3** |
| Threads (GCC and Clang) | RangePolicy | 8 | 1.25 | **8.75** | **+600%** | **6** | **7** |
| Threads (GCC and Clang) | MDRangePolicy, tile=1 | 1/2/4/8 | 1.25 | 1.25 | 0% | 0 | 1 |
| Threads (GCC and Clang) | MDRangePolicy, tile=4 | 1/2/4/8 | 1.25 | **5.00** | **+300%** | **3** | 1 |
| OpenMP (GCC) | MDRangePolicy, tile=1 | 1/2/4/8 | 1.25 | 1.25 | 0% | 0 | 1 |
| OpenMP (GCC) | MDRangePolicy, tile=4 | 1/2/4/8 | 1.25 | **5.00** | **+300%** | **3** | 1 |
| OpenMP (GCC) | RangePolicy | 1/2 | 1.25 | 1.25 | 0% | 0 | 1 |
| OpenMP (GCC) | RangePolicy | 4/8 | 1.25 | **TIMEOUT** | n/a | unobserved | n/a |

Every completed ordinary-range baseline of 32 elements returned the correct sum `40` with 32 calls. Normal-range multithreaded tests used multiple worker thread IDs (e.g. Threads 8 workers processed the baseline). The boundary tests involving a single tile have one worker, so those results do **not** imply cross-thread overlap.

For Threads the RangePolicy observations are identical in GCC and Clang and across all three repeats per thread configuration; no valid index was called twice, and the recorded `cross_thread_same_index` was 0. The extra calls are to *different invalid index values* and, for the 4/8-thread cases, happen on multiple distinct worker threads. That is an incorrect partitioning result, not direct evidence of concurrent writes to the same location.

For OpenMP, at 4 or 8 threads the high-origin one-element RangePolicy reduction exceeded the 5-second watchdog in all **12 of 12** configurations/repetitions (two tile parameters, three repetitions, two thread counts); all 1/2-thread controls returned correct numerical results. The precise cause of the timeout is not yet proven; notably `tile` is unused by the RangePolicy path, so it is not an independent change in that configuration. Do not interpret a timeout as confirmed deadlock or arithmetic overflow without stack/counter evidence.

## Meaning of percentage deviation

For an ideal scalar constant contribution `1.25`, the correct sum is `1.25`. A wrong sum of `3.75`, `5.00`, or `8.75` represents **+200%, +300%, or +600% error respectively**. These are constructed reduction examples, not fixed error bounds for real applications. If the invalid indices are used as array subscripts, the impact can instead include invalid memory access, corrupted output, or a crash; none is established by this recorder.

A numerically wrong reduction is a correctness defect independent of whether any data race exists. If a simulation is iterative, a single incorrect step can propagate; no historical scientific application output has been audited.

## Remaining gaps

1. Do not claim affected users must rerun past experiments without first matching the Kokkos version, signed index type, index location, tile configuration, backend, and numerical sensitivity. Inputs far from the integer boundary have not failed in these tests.
2. Establish real-memory overlap separately in a safe, bounded test with guards; record same index handled by multiple threads. Current experiments recorded **no** same-index cross-thread overlap.
3. Diagnose OpenMP 4/8-thread timeout with bounded instrumentation / backtraces and a minimal isolated repro; avoid calling it a deadlock yet.
4. Test patches in A/B configurations for `HostIterateTile` and `WorkRange` with both sanitizers and actual correctness assertions, plus normal workloads.
5. Verify active upstream main/head separately before submitting findings upstream; this pinned audit is reproducible against the stated commit and not necessarily the latest code.

## Workflow interpretation

The workflow returns success after collecting per-case exit codes; **this is NOT a statement that all cases passed**. The `numerical-results.csv` artifact and the job logs contain individual outcomes; the timeout watchdog reports exit code 124.
