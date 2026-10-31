# Finding 2: MDRangePolicy Host Tile boundary overrun

Date: 2026-10-31. Status: confirmed (public API, real kernel calls, two compilers).
No upstream changes and **no tests on user's Mac**.

## Invariant

For a `Kokkos::MDRangePolicy` with half-open bounds `[lower, upper)`,
every invocation of the user functor must have indices within those bounds,
independent of the selected tile size.

## Source

`core/src/impl/KokkosExp_Host_IterateTile.hpp`,
`HostIterateTile::check_iteration_bounds()`:

```cpp
if ((offset[i] + m_tile[i]) <= m_upper[i]) {
  partial_tile[i] = m_tile[i];
} else {
  partial_tile[i] = m_upper[i] - offset[i];
}
```

The addition is performed in `index_type`, which is signed 32-bit for
`IndexType<int32_t>`. For offsets near `INT_MAX`, the addition overflows.
The resulting "full tile" decision causes extra user functor calls.

## Reproducer

Using *only* public Kokkos APIs, `Kokkos::Serial`, rank 2:

```cpp
using P = Kokkos::MDRangePolicy<
    Kokkos::Serial, Kokkos::Rank<2>, Kokkos::IndexType<int32_t>>;
P p({INT_MAX - 1, 0}, {INT_MAX, 1}, {4, 1});
Kokkos::parallel_for(p, [&](int32_t i, int32_t j) { /* record i,j */ });
```

The logical iteration count is **1**; no large View or huge work range exists.

## GitHub Actions results

GitHub Actions run: https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41209083202

| Commit | Compiler | One-element baseline | Boundary functor calls | Out-of-range indices | UBSan |
|---|---|---|---:|---:|---|
| Pre-9650 `675433c879a1a19ecf48338a727d16b8bfbf8ff9` | GCC | PASS | 1 | 0 | PASS |
| Pre-9650 | Clang | PASS | 1 | 0 | PASS |
| Develop `1e08552799ab366f3037ce7ad46cc8d3ca85498d` | GCC | PASS | 4 | 3 | signed overflow |
| Develop | Clang | PASS | 4 | 3 | signed overflow |

Actual boundary indices from both optimized runs:
`2147483646, 2147483647, -2147483648, -2147483647`

UBSan source:
`impl/KokkosExp_Host_IterateTile.hpp:1221:22`,
`2147483646 + 4` cannot be represented in `int`.

The test records indices but does not perform out-of-range memory writes.
That is why memory-corruption and exploitability claims are **not** made.

## Meaning

The *declared* work length is not modified.
The incorrect "full tile" classification makes the *executed* work set
larger than the declared interval. `parallel_reduce` goes through the same
`HostIterateTile::check_iteration_bounds` method, although the actual
reduction output is not separately tested.

Candidate direction to investigate in a follow-up (not yet patched):
compare `m_tile[i]` against `m_upper[i] - offset[i]`, under the existing
invariant that the calculated offset lies within the user-specified interval.

## Deduplication note

The same risk was identified in GitHub Copilot review of upstream PR #9650 on
2026-10-14, then marked resolved after discussion, but the current pinned
develop version still reproduces it.
https://github.com/kokkos/kokkos/pull/9650#discussion_r4274643141

No upstream Issue, PR or patch was created.
