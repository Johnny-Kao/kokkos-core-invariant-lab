// PRIVATE K07 research: standard-sized public API for reusing BinSort.
// No invalid bin keys or undersized buffers; only 8 keys, two normal bins.
// Control constructs a new sorter each time. RED reuses one sorter as documented.
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <array>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using Exec = Kokkos::OpenMP;
using Keys = Kokkos::View<int*, Exec>;
using BinOp = Kokkos::BinOp1D<Keys>;
using Sorter = Kokkos::BinSort<Keys, BinOp>;

static void set_keys(Keys keys, int round) {
  Kokkos::parallel_for(
      "k07_set_normal_keys", Kokkos::RangePolicy<Exec>(0, 8),
      KOKKOS_LAMBDA(const int i) {
        keys(i) = round == 0 ? (i < 4 ? 0 : 10) : (i < 4 ? 10 : 0);
      });
  Exec{}.fence();
}

static void check_permutation(const Sorter& sorter, int round) {
  auto copy = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, sorter.get_permute_vector());
  for (int i = 0; i < 8; ++i) {
    int expected = round == 0 ? i : ((i + 4) % 8);
    if (copy(i) != static_cast<unsigned>(expected)) {
      std::fprintf(stderr,
                   "K07_PERM_MISMATCH round=%d index=%d got=%llu expected=%d\n",
                   round, i, static_cast<unsigned long long>(copy(i)), expected);
      std::exit(4);
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 2 || (std::strcmp(argv[1], "control") != 0 &&
                    std::strcmp(argv[1], "reuse") != 0)) {
    std::fprintf(stderr, "usage: k07_binsort_reuse control|reuse\n");
    return 2;
  }
  const bool reuse = std::strcmp(argv[1], "reuse") == 0;
  Kokkos::initialize();
  {
    Keys keys("eight_normal_keys", 8);
    BinOp bin_op(2, 0, 10);
    std::printf("K07_MODE=%s N=8 BINS=3 VALUES_IN_RANGE=[0,10]\n",
                reuse ? "reuse" : "control");
    std::fflush(stdout);
    if (reuse) {
      Sorter sorter(keys, bin_op, false);
      for (int round = 0; round < 2; ++round) {
        set_keys(keys, round);
        std::printf("K07_REUSE_BEGIN_ROUND=%d\n", round);
        std::fflush(stdout);
        sorter.create_permute_vector();
        check_permutation(sorter, round);
        std::printf("K07_REUSE_ROUND_PASS=%d\n", round);
        std::fflush(stdout);
      }
    } else {
      for (int round = 0; round < 2; ++round) {
        set_keys(keys, round);
        Sorter fresh(keys, bin_op, false);
        fresh.create_permute_vector();
        check_permutation(fresh, round);
        std::printf("K07_FRESH_ROUND_PASS=%d\n", round);
        std::fflush(stdout);
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
