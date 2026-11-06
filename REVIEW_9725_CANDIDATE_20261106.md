# Kokkos #9725 consolidated research candidate — 2026-11-06

Branch: `research/9725-review-consolidated-20261106`
Source pinned: `kokkos/kokkos@1e08552799ab366f3037ce7ad46cc8d3ca85498d`
Research validation: https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/42198620881
**No changes to the upstream PR or staging branch.**

## Scope included (Nov 3 review consensus plus Nov 5 follow-up)
- Preserve the original overflow-safe default tile recommendation comparison.
- Replace both the default tile selection and `else`-branch shrinking comparisons with division-based guards; the `else` case was explicitly flagged on Nov 5.
- Guard actual tile product **before** narrowing to `int` and multiplication; reject excess tiles intentionally.
- Use overflow-free quotient/remainder ceiling division for `m_tile_end`, required for the new no-explicit-tile regression.
- Run the probe with `Kokkos::DefaultHostExecutionSpace` (Serial not assumed).
- Preserve equality at the cap (`<=` rather than `<`).

The exact isolated implementation is generated as the `final_9725` header variant in `review_audit/build_header_variants.py`. It matches the prior `complete_arithmetic` variant, now with an explicit final candidate name. No HostIterateTile runtime changes are included.

## Strict CI
The `review-suggestion-ab.yml` workflow now compiles and executes only the candidate, on:
- GCC/Serial, Clang/Serial, GCC/OpenMP-only, GCC/Threads-only
- Each in optimized and signed-overflow UBSan builds
- Six expected-success cases and two expected defensive-rejection cases per build.
- Unexpected success, timeout, error message mismatch or UBSan overflow fails the job.

Result: **4 of 4 jobs passed** (2026-11-06, run 42198620881). Logs were checked for real numerical results and intentional rejection messages, not merely green run indicators.

## Staging gate not yet satisfied
- Isolated installed-header probe, not native Kokkos unit-test integration.
- `DefaultHostExecutionSpace` is used by research probes, but the actual upstream unit-test patch still needs rewriting.
- No CUDA, HIP, SYCL or NVIDIA GH200 compilation tested.
- No staging branch or upstream #9725 modification.
- Preserve separate future tasks: widening bookkeeping integer types, HostIterateTile high-origin numerical correctness, and Threads WorkRange problems.

**Next decision:** if approved, synthesize a clean two-file upstream-compatible patch in staging and run native Kokkos tests and UBSan before touching #9725.
