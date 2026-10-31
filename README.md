# Kokkos Core Invariant Lab

Independent research on Kokkos core correctness invariants. No changes are applied to the upstream repository. **All compilation and execution occur in GitHub Actions.**

## Candidate: `MDRangePolicy` signed integer overflow in tile ceil division

Source: `core/src/KokkosExp_MDRangePolicy.hpp`, `update_tiling_properties()`.

The constructor checks that the **product of range lengths** fits `index_type`. Its tile count formula nevertheless evaluates `(length + tile_size - 1) / tile_size` in that same index type.

This probe constructs a legal `MDRangePolicy<Rank<2>, IndexType<int32_t>>` with bounds `{0,0}..{INT32_MAX,1}` and tile `{2,1}`. Its entire work range fits signed 32-bit, and its correct tile count `(int64_t(INT32_MAX)+1)/2` also fits signed 32-bit. The *intermediate addition* does not.

- Baseline: 1,023 iterations and 512 tiles; expected success.
- Boundary: INT32_MAX iterations, 1,073,741,824 tiles; should also be constructible without overflow.
- Build with signed integer overflow sanitizer, so the test catches the first UB in instantiated headers.

Kokkos source pinned to `1e08552799ab366f3037ce7ad46cc8d3ca85498d` for reproducibility.

**Research status:** candidate only until GitHub Actions results are examined. This does **not** claim an out-of-bounds write or exploitable security vulnerability.

### Testing

Open the Actions tab, select `MDRange Invariant`, inspect both GCC and Clang jobs. This repo does not contain a local test script.
