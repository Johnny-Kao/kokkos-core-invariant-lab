// Read-only Kokkos Core arithmetic audit. No billion-iteration kernels are launched.
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

using Index = std::int64_t;
using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<Index>>;

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const std::string mode(argv[1]);
    const Index maximum = std::numeric_limits<Index>::max();
    if (mode == "baseline-auto") {
      Policy policy(Index(0), Index(1024));
      std::cout << "BASELINE_AUTO chunk=" << policy.chunk_size() << "\n";
      if (policy.chunk_size() <= 0) rc = 3;
    } else if (mode == "boundary-auto") {
      // Public constructor: exercises automatic chunk selection, no work execution.
      Policy policy(Index(0), maximum);
      std::cout << "BOUNDARY_AUTO chunk=" << policy.chunk_size() << "\n";
      if (policy.chunk_size() <= 0) rc = 4;
    } else if (mode == "baseline-workrange") {
      // Explicit ChunkSize bypasses auto-chunk computation.
      Policy policy(Index(0), Index(1024), Kokkos::ChunkSize(1));
      Policy::WorkRange part(policy, 1, 2);
      std::cout << "BASELINE_WORKRANGE " << part.begin() << " " << part.end() << "\n";
      if (part.begin() != 512 || part.end() != 1024) rc = 5;
    } else if (mode == "boundary-workrange") {
      // Legal public bounds and a small partition count, but ceil-rounding
      // intermediate arithmetic may overflow signed Index.
      Policy policy(Index(0), maximum, Kokkos::ChunkSize(1));
      Policy::WorkRange part(policy, 0, 2);
      std::cout << "BOUNDARY_WORKRANGE " << part.begin() << " " << part.end() << "\n";
      if (part.begin() < 0 || part.end() < part.begin()) rc = 6;
    } else if (mode == "small-workrange-high-origin") {
      // One actual logical element: tests whether high absolute index alone overflows.
      using SmallPolicy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::int32_t>>;
      const std::int32_t hi = std::numeric_limits<std::int32_t>::max();
      SmallPolicy policy(hi - 1, hi, Kokkos::ChunkSize(1));
      SmallPolicy::WorkRange part(policy, 1, 2);
      std::cout << "SMALL_HIGH_ORIGIN " << part.begin() << " " << part.end() << "\\n";
      if (part.begin() != hi || part.end() != hi) rc = 8;
    } else if (mode == "boundary-negative-origin") {
      Policy policy(Index(-1), maximum, Kokkos::ChunkSize(1));
      Policy::WorkRange part(policy, 0, 2);
      std::cout << "NEGATIVE_ORIGIN_WORKRANGE " << part.begin() << " " << part.end() << "\n";
      if (part.begin() < -1 || part.end() < part.begin()) rc = 7;
    } else {
      rc = 2;
    }
  }
  Kokkos::finalize();
  return rc;
}
