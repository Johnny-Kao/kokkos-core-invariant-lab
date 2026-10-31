# Finding 3 — Pre-existing integer overflow in CPU TileSizeRecommended

**Status:** Proven using public `MDRangePolicy` construction on GitHub Actions.
**Scope:** Original upstream source pinned before and after #9650; no upstream changes.

## Contract and cause

Kokkos permits explicitly specified tile dimensions, and the Host
backends have no extra hardware tile-size restriction. For Host,
`max_total_tile_size` is `INT_MAX`. An `IndexType<int32_t>` range
of shape `{1, INT_MAX}` has total logical extent `INT_MAX` and
is valid under the flat-range constraint.

Use **explicitly small tiles**: `{1,2}`.

```cpp
using P = Kokkos::MDRangePolicy<Kokkos::Serial, Kokkos::Rank<2>,
                                Kokkos::IndexType<std::int32_t>>;
P p({0, 0}, {1, INT32_MAX}, {1, 2});
```

The constructor calls `tile_size_recommended()` even if all tile sizes
are supplied. CPU's generic `TileSizeRecommended::get` uses:

```cpp
int prod_tile_size = 1;
// the innermost dimension recommends its full length (INT_MAX)
if (prod_tile_size * default_tile_size <= max_total_tile_size) {
    ...
}
```

Once `prod_tile_size == INT_MAX`, testing the next rank multiplies it
by the default 2, overflowing signed 32-bit `int` *before* the branch
can reject the extra tile. The actual requested tile product is just 2,
well below the permitted maximum.

## Before and after #9650

GitHub Actions:
https://github.com/Johnny-Kao/kokkos-core-invariant-lab/actions/runs/41213188537

All 4 independent pinned-source configurations (GCC and Clang, pre #9650
`675433c879a1a19ecf48338a727d16b8bfbf8ff9`, and post #9650
`1e08552799ab366f3037ce7ad46cc8d3ca85498d`) reproduce the same
UBSan finding: `2147483647 * 2` overflows signed `int`.

Both compiler/revision combinations succeed with the small 128-element
control. No large allocation or kernel execution.

**Conclusion:** an independent, *pre-existing* recommendation bug, not
a regression introduced by #9650.

## Related upstream work

- https://github.com/kokkos/kokkos/issues/9266 addresses default CPU
  tile-size strategy/performance, not this verified integer overflow.
- https://github.com/kokkos/kokkos/issues/8103 addresses an unrelated
  CUDA tile size / execution problem; not the same arithmetic defect.

Initial minimal repair concept: replace potentially overflowing
`prod_tile_size * default_tile_size <= max_total_tile_size` with a
bounded positive-integer comparison
`prod_tile_size <= max_total_tile_size / default_tile_size`.
This is a direction only, not yet patched or tested.

The separate `index_type=int64_t` conversions in this function may
require additional consideration beyond the specific confirmed
`int32_t` reproducer.

No changes to the upstream repository or staging fix branch.
