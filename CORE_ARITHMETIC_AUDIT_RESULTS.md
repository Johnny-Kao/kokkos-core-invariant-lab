# Core Arithmetic Audit — First Dynamic Results (2026-11-03)

Repository: Johnny-Kao/kokkos-core-invariant-lab
Branch: research/core-arithmetic-audit-20261103
Source pinned: kokkos/kokkos commit 1e08552799ab366f3037ce7ad46cc8d3ca85498d
Workflow: https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41616066202
GCC job: https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41616066202/job/125849701642
Clang job: https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41616066202/job/125849702071

## What was executed

Build pinned original Kokkos Serial with both compilers and run a real public RangePolicy construction / nested WorkRange probe with `-fsanitize=signed-integer-overflow -fno-sanitize-recover=signed-integer-overflow`. There is **no large kernel execution or allocation**.

Note that the GitHub workflow conclusion is success by design: individual diagnostic exits are recorded separately and do not fail the entire aggregation job. **Do not interpret workflow success as all tests passing.**

| Scenario | GCC status | Clang status | Evidence |
|---|---|---|---|
| baseline-auto (0..1024) | exit 0 | exit 0 | chunk=32 |
| baseline-workrange (0..1024, 2 partitions) | exit 0 | exit 0 | [512,1024) |
| boundary-auto (0..INT64_MAX) | exit 1, UBSan | exit 1, UBSan | Kokkos_ExecPolicy.hpp:241 signed multiply |
| boundary-workrange (0..INT64_MAX, explicit ChunkSize(1), 2 partitions) | exit 1, UBSan | exit 1, UBSan | Kokkos_ExecPolicy.hpp:334 signed add |
| boundary-negative-origin (-1..INT64_MAX, explicit ChunkSize(1), 2 partitions) | exit 1, UBSan | exit 1, UBSan | Kokkos_ExecPolicy.hpp:334 signed subtract |

## Confirmed arithmetic failures

- C-01: `new_chunk_size * 100`: `144115188075855872 * 100` does not fit signed int64. Both compilers, source line 241.
- C-02a: `INT64_MAX + 1` during work-part ceil calculation. Both compilers, source line 334.
- C-02b: `INT64_MAX - (-1)` during range length calculation. Both compilers, source line 334.

## Limitations and remaining work

- A confirmed arithmetic UB is **not** automatically a proven end-user wrong result or security vulnerability.
- `WorkRange` is instantiated directly through its publicly accessible nested type; validate whether normal production scheduler paths reach the same expression with these boundary inputs.
- Do not claim a numerical wrong output beyond UBSan until a release-like result is inspected.
- C-03 HostThreadTeam, C-04 SYCL Range, C-05 ScratchSpace, C-06 SYCL MDRange, C-07 CopyViews have **not** been dynamically tested in this run.
- No change to upstream PR #9725 and no upstream issue created.
- Next gate: production reachability, additional precondition checks, isolated A/B fixes, and backend-specific tests.

## Additional result: ONE-ELEMENT high-origin WorkRange overflow

Actions: https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41617155944
GCC and Clang both reproduced `Kokkos_ExecPolicy.hpp:339` signed integer overflow.

Public bounds with **one** logical element: `RangePolicy<Serial, IndexType<int32_t>>(INT32_MAX-1, INT32_MAX, ChunkSize(1))`.
Construct the publicly accessible `Policy::WorkRange(policy, 1, 2)`. Expected partition after the end is empty `[INT32_MAX, INT32_MAX)`. The implementation computes `m_begin + work_part` before clamping the resulting `m_end` to the policy upper bound; this evaluates `INT32_MAX + 1`, triggering UBSan (both compilers, exit 1).

**Production reachability:** `WorkRange` is called by real Threads parallel_for / parallel_reduce and by selected CUDA/HIP/HPX backend paths, but this particular one-element probe instantiated the nested WorkRange directly using a Serial policy. Thus the arithmetic bug is confirmed, while live Threads/CUDA/HIP execution with one element has **not** yet been dynamically demonstrated. Do not conflate direct WorkRange invocation with confirmed backend scheduling corruption.

Interpretation: large iteration count is not a necessary condition for this intermediate overflow; a small logical range with a high absolute origin can trigger it. No incorrect actual functor invocation has been demonstrated by this specific experiment.
