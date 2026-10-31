# Finding: MDRangePolicy tile-count overflow (2026-10-31)

## Independent discovery

Source-first audit identified `core/src/KokkosExp_MDRangePolicy.hpp` within `MDRangePolicy::update_tiling_properties()`:

```cpp
this->m_tile_end[i] = static_cast<index_type>(
    (length + this->m_tile[i] - 1) / this->m_tile[i]);
```

Range validation uses checked multiplication of extents and rejects work-domain products that exceed `index_type::max()`. It does **not** guarantee that adding `tile_size - 1` to an individual range length fits `index_type`.

## Public API reproducer

```cpp
using Policy = Kokkos::MDRangePolicy<
    Kokkos::Rank<2>, Kokkos::IndexType<std::int32_t>>;
Policy policy({0, 0}, {INT32_MAX, 1}, {2, 1});
```

- Logical work extent: `INT32_MAX * 1` (fits signed 32-bit).
- Legal tile: `{2, 1}`.
- Expected `impl_num_tiles()`: `1073741824` (fits signed 32-bit).
- Observed under optimized GCC and Clang: `-1073741824`.
- UBSan in both compilers: signed overflow at `KokkosExp_MDRangePolicy.hpp:563:19`, specifically `2147483647 + 2`.
- Baseline with `length=1023` returns correct 512 tiles.

No giant allocation or execution of billions of work items is required: constructing and inspecting the policy suffices.

## Regression comparison

Merged upstream [kokkos/kokkos#9650](https://github.com/kokkos/kokkos/pull/9650) changed `point_type` and `tile_type` from `int64_t` to `index_type` (merged 2026-10-27). The same probe on the PR's pre-merge base commit has no UB and gives the correct result.

| Source | GCC UBSan | GCC optimized result | Clang UBSan | Clang optimized result |
| --- | --- | --- | --- | --- |
| Pre-9650 `675433c879a1a19ecf48338a727d16b8bfbf8ff9` | PASS | 1073741824 | not tested | not tested |
| Develop `1e08552799ab366f3037ce7ad46cc8d3ca85498d` | overflow | -1073741824 | overflow | -1073741824 |
| Pinned develop plus candidate patch | PASS | 1073741824 | PASS | 1073741824 |

## Minimal candidate repair

Replace the overflow-prone ceil-division numerator with:

```cpp
length / tile + (length % tile != 0)
```

The candidate patch is under `candidate/mdrange-ceil-no-overflow.patch`. It is applied **only inside Actions**; upstream is untouched. Positive tile sizes are already selected/required at the relevant point in construction.

## Actions evidence

- Failing pinned develop, sanitizer and wrong output: https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41204784155
- Passing pre-9650 base: https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41204984592
- Passing pinned develop with candidate patch: https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41205107457

## Scope and limitations

This confirms a signed overflow and wrong computed tile count in a valid public API path for large 32-bit index ranges. It does not prove memory corruption, practical exploitability, or wrong results in a real GPU kernel. The reproduced boundary is large (2.15 billion work items), though legitimate in a high-performance scientific application. There was no GPU testing. Only targeted GitHub Issue/PR searches for the exact tile-count overflow were conducted; broader duplicate vetting should precede upstream filing. No upstream Issue or PR was created.
