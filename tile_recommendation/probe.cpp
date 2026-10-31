#include <Kokkos_Core.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>

// A valid two-dimensional range with explicitly supplied, small tile sizes.
// No kernel is executed and no large allocation is performed.
int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  const bool boundary = argc > 1 && std::string_view(argv[1]) == "--boundary";
  const std::int32_t n = boundary ? std::numeric_limits<std::int32_t>::max() : 128;

  using Policy = Kokkos::MDRangePolicy<Kokkos::Serial, Kokkos::Rank<2>,
                                       Kokkos::IndexType<std::int32_t>>;
  std::cout << "boundary=" << boundary
            << " declared_extent=" << n
            << " requested_tile={1,2}" << std::endl;

  // The innermost dimension is length n, with an outer length of 1.
  // The requested tile is small; the internal recommender is still called.
  Policy policy({0, 0}, {1, n}, {1, 2});

  std::cout << "constructed; declared_extent=" << n
            << " num_tiles=" << policy.impl_num_tiles() << std::endl;
  return 0;
}
