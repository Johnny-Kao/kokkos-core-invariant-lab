#include <Kokkos_Core.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>

// The one-dimensional logical interval is deliberately very short (1 element)
// even when its absolute starting index is close to INT32_MAX.
// No large View, allocation, or high iteration count is required.
int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);

  const bool boundary = argc > 1 && std::string_view(argv[1]) == "--boundary";
  const std::int32_t lower =
      boundary ? std::numeric_limits<std::int32_t>::max() - 1 : 100;
  const std::int32_t upper = lower + 1;  // safe for both cases
  const std::int32_t tile = 4;

  using Policy = Kokkos::MDRangePolicy<Kokkos::Serial, Kokkos::Rank<2>,
                                       Kokkos::IndexType<std::int32_t>>;
  Policy policy({lower, 0}, {upper, 1}, {tile, 1});

  int count = 0;
  int wrong = 0;
  std::int32_t samples[8] = {};
  Kokkos::parallel_for("short_interval_near_index_max", policy,
                       [&](std::int32_t i, std::int32_t j) {
                         if (count < 8) samples[count] = i;
                         ++count;
                         if (i < lower || i >= upper || j != 0) ++wrong;
                       });
  Kokkos::fence();

  std::cout << "boundary=" << boundary << " lower=" << lower
            << " upper=" << upper << " tile=" << tile
            << " declared_length=" << (std::int64_t{upper} - lower)
            << " calls=" << count << " invalid_indices=" << wrong
            << " indices=";
  for (int k = 0; k < count && k < 8; ++k) {
    if (k) std::cout << ',';
    std::cout << samples[k];
  }
  std::cout << '\n';

  if (count != 1 || wrong != 0 || samples[0] != lower) {
    std::cerr << "INVARIANT VIOLATION: kernel invoked outside declared interval\n";
    return 2;
  }
  return 0;
}
