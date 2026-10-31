#include <Kokkos_Core.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  const bool boundary = argc > 1 && std::string_view(argv[1]) == "--boundary";
  const std::int32_t length =
      boundary ? std::numeric_limits<std::int32_t>::max() : 1023;

  using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>,
                                       Kokkos::IndexType<std::int32_t>>;
  // Total logical iterations = length * 1, which fits int32_t.
  // Each tile covers two iterations along the first dimension.
  Policy policy({0, 0}, {length, 1}, {2, 1});

  const std::int64_t expected = (std::int64_t{length} + 1) / 2;
  const std::int64_t actual = policy.impl_num_tiles();
  std::cout << "length=" << length << " expected_tiles=" << expected
            << " actual_tiles=" << actual << '\n';

  if (actual != expected) {
    std::cerr << "INVARIANT VIOLATION: valid range but wrong tile count\n";
    return 2;
  }
  return 0;
}
