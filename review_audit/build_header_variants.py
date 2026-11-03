#!/usr/bin/env python3
"""Generate non-upstream Kokkos header variants for reviewer suggestion A/B CI.

Only modifies copies of headers in the GitHub Actions runner.
Fails closed if source text changes, so results cannot silently run unpatched.
"""
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1])
dest = Path(sys.argv[2])
source = root / "include/KokkosExp_MDRangePolicy.hpp"
original = source.read_text()

def subst(src, old, new, desc):
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"{desc}: expected one occurrence, found {n}")
    return src.replace(old, new)

pr_old = "if (prod_tile_size * default_tile_size <= max_total_tile_size) {"
pr_new = "if (prod_tile_size <= max_total_tile_size / default_tile_size) {"
review_a_old = """if (static_cast<tile_size_type>(this->m_prod_tile_dims) *
                default_tile[i] <=
            effective_max_tile_size) {"""
review_a_new = """if (static_cast<tile_size_type>(this->m_prod_tile_dims) <=
            effective_max_tile_size / default_tile[i]) {"""
review_b_old = """static_cast<tile_size_type>(this->m_prod_tile_dims) *
                         default_tile[i] >
                     effective_max_tile_size) {"""
review_b_new = """static_cast<tile_size_type>(this->m_prod_tile_dims) >
                     effective_max_tile_size / default_tile[i]) {"""
review_actual_old = """      this->m_prod_tile_dims *= static_cast<int>(this->m_tile[i]);"""
review_actual_new = """      // Check before narrowing or multiplying (actual tile product is int).
      // The guards use the already established positive tile sizes.
      if (this->m_prod_tile_dims <= 0 ||
          static_cast<std::uintmax_t>(this->m_tile[i]) >
              static_cast<std::uintmax_t>(this->m_max_total_tile_size) /
                  static_cast<std::uintmax_t>(this->m_prod_tile_dims)) {
        Kokkos::abort("Kokkos::MDRangePolicy tile dimensions exceed int limit");
      }
      this->m_prod_tile_dims *= static_cast<int>(this->m_tile[i]);"""

def make_variant(name, contents):
    variant = dest / name / "include"
    shutil.copytree(root / "include", variant, dirs_exist_ok=True)
    (variant / "KokkosExp_MDRangePolicy.hpp").write_text(contents)
    print(name, "header written")

make_variant("original", original)
pr = subst(original, pr_old, pr_new, "PR recommendation guard")
make_variant("pr_only", pr)
reviewed = subst(pr, review_a_old, review_a_new, "review default selection")
reviewed = subst(reviewed, review_b_old, review_b_new, "review default shrinking")
reviewed = subst(reviewed, review_actual_old, review_actual_new, "review actual tile product guard")
make_variant("reviewed", reviewed)
print("All 3 variants use original pinned Kokkos headers, PR suggestion and isolated reviewed candidate.")

# The reviewer's no-explicit-tile test also reaches this independent
# addition overflow. Only the 'complete_arithmetic' candidate includes it.
ceil_old = """      this->m_tile_end[i] = static_cast<index_type>(
          (length + this->m_tile[i] - 1) / this->m_tile[i]);"""
ceil_new = """      this->m_tile_end[i] = static_cast<index_type>(
          length / this->m_tile[i] + (length % this->m_tile[i] != 0));"""
complete = subst(reviewed, ceil_old, ceil_new, "ceil division without overflow")
make_variant("complete_arithmetic", complete)
