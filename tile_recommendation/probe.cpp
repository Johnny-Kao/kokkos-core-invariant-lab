#include <Kokkos_Core.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>

// No kernel or large memory allocation is needed for this public-API check.
int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  const bool boundary = argc > 1 && std::string_view(argv[1]) == "--boundary";
  // INT_MAX - 1 keeps the unrelated (length + tile - 1) expression
  // representable for explicitly requested tile size one.
  const std::int32_t n =
      boundary ? std::numeric_limits<std::int32_t>::max() - 1 : 128;

  using Policy = Kokkos::MDRangePolicy<Kokkos::Serial, Kokkos::Rank<2>,
                                       Kokkos::IndexType<std::int32_t>>;
  std::cout << "boundary=" << boundary << " extent=" << n
            << " explicit_tile={1,1}" << std::endl;

  // Legal logical extent = n. The default recommendation, though unused,
  // still attempts (n * 2) in signed int for the outer dimension.
  Policy policy({0, 0}, {1, n}, {1, 1});
  auto const recommended = policy.tile_size_recommended();
  const auto actual_tiles = policy.impl_num_tiles();

  std::cout << "num_tiles=" << actual_tiles << " recommended={"
            << recommended[0] << ',' << recommended[1] << "}" << std::endl;

  if (actual_tiles != n || (boundary && (recommended[0] != 1 ||
                                         recommended[1] != n))) {
    std::cerr << "INVARIANT VIOLATION: valid explicit-tile MDRange policy"
              << std::endl;
    return 2;
  }
  return 0;
}
