# Kokkos Core — Cross-module intermediate arithmetic audit

Date: 2026-11-03 (JST)
Status: **dynamic verification running; findings below are source-level candidates unless linked to earlier completed evidence**.
Research branch: `research/core-arithmetic-audit-20261103`
Actions: https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41616066202
Pinned upstream source: `1e08552799ab366f3037ce7ad46cc8d3ca85498d`.
No upstream PR modifications. Test execution occurs in GitHub Actions only.

## Already reproduced in this research repository

1. `MDRangePolicy` tile ceil division overflow — `FINDINGS.md`; UBSan and incorrect tile count with valid inputs.
2. `HostIterateTile::check_iteration_bounds` overflow — `TILE_BOUNDARY_FINDING.md`; one valid logical element yields four functor calls, three outside bounds.
3. Host `TileSizeRecommended` multiplication overflow — `TILE_RECOMMENDATION_FINDING.md`; reproduced before and after PR #9650.

## New cross-module candidates (NOT yet confirmed)

| Candidate | Location on upstream develop | Arithmetic to examine | Question |
| --- | --- | --- | --- |
| C-01 RangePolicy auto chunk | `core/src/Kokkos_ExecPolicy.hpp:240-249` | `new_chunk_size * 100 * concurrency`, `m_end - m_begin` | Can legal public bounds trigger signed overflow before comparison? |
| C-02 RangePolicy WorkRange | `core/src/Kokkos_ExecPolicy.hpp:334-339` | `(end-begin)+(part_size-1)`, then `work_part * part_rank` | Can a valid range and partition produce wrong bounds? |
| C-03 HostThreadTeam partition | `core/src/impl/Kokkos_HostThreadTeam.hpp:327-352` | `length + INT_MAX`, `m_work_end + chunk - 1` | Are maximal lengths reachable from production calls? |
| C-04 SYCL Range launch round-up | `core/src/SYCL/Kokkos_SYCL_ParallelFor_Range.hpp:120-143` | `actual_range + workgroup_size - 1` | Does `size_t` wrap before launch bounds check? |
| C-05 ScratchSpace allocation | `core/src/Kokkos_ScratchSpace.hpp:87-97` | `size * m_multiplier` before capacity check | Are sizes bounded before unsigned multiplication? |
| C-06 SYCL MDRange launch comparison | `core/src/SYCL/Kokkos_SYCL_ParallelFor_MDRange.hpp:161-211` | `max_grid_size * local_range`, `m_extent[0] * m_extent[1]` | Can overflow invert grid-stride selection? |
| C-07 CopyViews byte count | `core/src/Kokkos_CopyViews.hpp:2492` | `sizeof(value_type) * dst.span()` | Are allocatable spans always bounded by byte capacity? |

## Current test batch

The new `core_arithmetic_audit.cpp` and `core-arithmetic-audit.yml` execute:
- Baseline auto chunk
- Baseline WorkRange
- INT64_MAX public RangePolicy auto chunk
- INT64_MAX explicit-chunk WorkRange
- Negative-origin boundary WorkRange

Each case runs under GCC and Clang with signed-overflow UBSan. Nonzero diagnostic exit codes are captured as data. Compilation failures must be distinguished from arithmetic failures. **A source-level candidate must not be labeled confirmed until execution logs demonstrate the specific arithmetic failure and the public/production reachability is validated.**

## Next validation gate

1. Inspect actual Actions logs and extract failing expression, compiler, and source line.
2. Validate real production call paths / preconditions and run an ordinary-range control.
3. For proven issues, build isolated minimal patches and run A/B with the same pinned source.
4. Handle HostThreadTeam, ScratchSpace, and SYCL separately after proving testability on permitted CI runners.
5. Keep cross-module changes separate from upstream MDRangePolicy #9725 until maintainers agree on scope.
