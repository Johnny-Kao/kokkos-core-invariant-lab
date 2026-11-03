# Upstream #9725 Review Suggestions — Independent A/B Evaluation
Date: 2026-11-03 JST
Branch: `research/core-arithmetic-audit-20261103`
Pinned upstream source: `kokkos/kokkos@1e08552799ab366f3037ce7ad46cc8d3ca85498d`.
No upstream PR modification. Builds/tests on GitHub Actions, not the local Mac.

## Reproducible execution
- Four-backend matrix including Serial/GCC, Serial/Clang, OpenMP/GCC (Serial **disabled**), Threads/GCC (Serial **disabled**).
- GCC/Clang, optimized and signed-overflow UBSan.
- Source variants compiled with isolated installed-header overlays, validated using exact-string replacements that abort on upstream source drift.
- [Review A/B with full construction and execution suite](https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41644028006)
- [No-Serial backend build coverage](https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41643851553)
- Existing [cross-thread numerical correctness](https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41619708733).

Important: the CI harness deliberately continues after diagnostic failures and intentional `Kokkos::abort()`; a green run is not equivalent to all test cases passing.

## Candidate versions

| Variant | Changes | Observation |
| --- | --- | --- |
| original | Original upstream | Host recommendation product multiplication overflows; no-explicit tile has invalid tile-count |
| pr_only | Only #9725 division comparison | Explicit tile recommendation fixed; no-explicit reviewer case still exhibits UBSan signed overflow and computed zero tile count |
| reviewed | pr_only + division guards for default tile selection/shrinking + early safe actual tile product guard | Explicit over-limit and too-wide tile abort before signed multiply, but no-explicit reviewer case still fails |
| complete_arithmetic | reviewed + overflow-free ceiling division in `m_tile_end` | Reviewer no-explicit case now passes and produces 1 tile; ordinary and explicit tests also pass, no signed overflow in these construction probes |
| complete_plus_host | complete_arithmetic + separate HostIterateTile execution bound comparison | Isolated runtime candidate; verify separately, not a change approved by reviewers |

## Individual reviewer concerns

1. **Kokkos::Serial is not always compiled:** Using `Kokkos::DefaultHostExecutionSpace` in research probes built and passed in OpenMP-only and Threads-only configurations with Serial disabled. This supports the test-code suggestion, but a real NVIDIA-GH200/CUDA CI build remains untested.
2. **Default tile recommendation multiply:** The one-line PR fix `prod <= cap / default` passes its original repro on both compilers. Guard positive default divisor as a precondition.
3. **Explicit tile product multiplication:** Without an early guard, `46341*46341` overflows signed 32-bit; the reviewed candidate converts it to a clear abort before multiplication (result 134 on Linux), including for an `int64_t` tile value wider than `int`. An abort is deliberate defensive rejection, not successful execution.
4. **Reviewer no-explicit-tile test:** `policy({0,0},{1,INT32_MAX-1})` is a useful regression. Original and reviewed-only variants produce zero tiles / UBSan. The independently added overflow-free `length / tile + (length % tile != 0)` is additionally required to make this case correct.
5. **Comparison semantics:** Reviewer suggested `prod < cap / next`. To preserve current `prod*next <= cap`, the mathematically equivalent replacement is `prod <= cap / next` for positive `next`. Using strict `<` changes exact-boundary acceptance. Explicit synthetic equality case `512*2==1024` verifies this.
6. **Type width:** Reviewer proposes `size_t`, 64-bit or `index_type`. It is not yet demonstrated that changing every type is needed or safe for all devices. Current internal `m_prod_tile_dims` is `int`: conversion to `int` before guards can silently narrow wider `index_type` tiles. A widened type redesign should be evaluated separately, with GPU and type/ABI implications considered.
7. **Actual parallel execution remains independent:** On the `complete_arithmetic` variant the real MDRange `parallel_reduce` with a single valid high-index point and explicit tile=4 still reports **4 instead of 1**, with 3 out-of-range functor calls. Under UBSan: `KokkosExp_Host_IterateTile.hpp:1221`. Tile=1 control returns 1. This is not fixed by reviewer constructor suggestions.
8. **Performance (screening only):** 100,000 normal policy constructions per variant were measured on shared Actions runners. Results are millisecond-scale with variable compiler/backend and scheduling noise; these **do not establish stable performance equivalence**. Need dedicated controlled repetitions before performance conclusions.

## Proposed contribution scope

Do **not** immediately apply every GitHub inline suggestion verbatim to upstream. Prefer:
- In current #9725: preserve its existing validated one-line fix; optionally include the two closely related safe default-tile comparisons and a narrowly targeted no-explicit-tile regression **if paired with the extra safe ceil-div change** and validated as a coherent patch.
- Early explicit tile product guard changes behavior and may deserve a separate PR/issue if maintainer wants narrow scope. The reviewer specifically says the broader `int` type redesign may be separate.
- Separate independent correctness issue: HostIterateTile runtime overflow causes wrong numerical results, even after the reviewed constructor arithmetic candidate.
- Separate independent correctness issue: Threads RangePolicy WorkRange causes wrong reduction and OpenMP high-origin RangePolicy reduction has a reproducible watchdog timeout. This is not in the #9725 diff.

## Open gates and caveats

- HostTile isolated runtime patch A/B is being run in a separate CI revision; only claim verified after reading its completed logs.
- Latest CUDA/HIP/SYCL compilation/runtime, 64-bit extreme index coverage, floating-point heterogeneous reductions, TSAN overlap/race assessment, long-run multithread nondeterminism, and formal maintainer agreement remain outstanding.
- No historic scientific result has been traced to this input-specific bug. Do not recommend blanket recomputation without matching affected version/backend/index/tile choices.
