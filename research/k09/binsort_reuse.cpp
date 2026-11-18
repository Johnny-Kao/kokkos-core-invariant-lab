// K09 research: standard-sized public API for reusing BinSort.
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
      "k09_set_normal_keys", Kokkos::RangePolicy<Exec>(0, 8),
      KOKKOS_LAMBDA(const int i) {
        keys(i) = round == 0 ? (i < 4 ? 0 : 10) : (i < 4 ? 10 : 0);
      });
  Exec{}.fence();
}

static void check_permutation(const Sorter& sorter, int round) {
  auto copy = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, sorter.get_permute_vector());
  // BinSort need not retain stable order for equal keys. Check the
  // permutation is bijective and ordered by each element's current key.
  // This is the correct oracle for sort_within_bins=false here:
  // every bin contains only one distinct key value.
  bool seen[8] = {};
  for (int i = 0; i < 8; ++i) {
    const auto raw = copy(i);
    if (raw >= 8 || seen[raw]) {
      std::fprintf(stderr,
                   "K09_INVALID_PERM round=%d pos=%d got=%llu duplicate_or_oob\n",
                   round, i, static_cast<unsigned long long>(raw));
      std::exit(4);
    }
    seen[raw] = true;
    const bool low_index = raw < 4;
    const bool expected_low_index = round == 0 ? i < 4 : i >= 4;
    if (low_index != expected_low_index) {
      std::fprintf(stderr,
                   "K09_INVALID_PERM round=%d pos=%d got=%llu wrong_bin\n",
                   round, i, static_cast<unsigned long long>(raw));
      std::exit(4);
    }
  }
  for (int i = 0; i < 8; ++i)
    if (!seen[i]) std::exit(4);
}

int main(int argc, char** argv) {
  if (argc != 2 || (std::strcmp(argv[1], "control") != 0 &&
                    std::strcmp(argv[1], "reuse") != 0)) {
    std::fprintf(stderr, "usage: k09_binsort_reuse control|reuse\n");
    return 2;
  }
  const bool reuse = std::strcmp(argv[1], "reuse") == 0;
  Kokkos::initialize();
  {
    Keys keys("eight_normal_keys", 8);
    BinOp bin_op(2, 0, 10);
    std::printf("K09_MODE=%s N=8 BINS=3 VALUES_IN_RANGE=[0,10]\n",
                reuse ? "reuse" : "control");
    std::fflush(stdout);
    if (reuse) {
      Sorter sorter(keys, bin_op, false);
      for (int round = 0; round < 2; ++round) {
        set_keys(keys, round);
        std::printf("K09_REUSE_BEGIN_ROUND=%d\n", round);
        std::fflush(stdout);
        sorter.create_permute_vector();
        check_permutation(sorter, round);
        std::printf("K09_REUSE_ROUND_PASS=%d\n", round);
        std::fflush(stdout);
      }
    } else {
      for (int round = 0; round < 2; ++round) {
        set_keys(keys, round);
        Sorter fresh(keys, bin_op, false);
        fresh.create_permute_vector();
        check_permutation(fresh, round);
        std::printf("K09_FRESH_ROUND_PASS=%d\n", round);
        std::fflush(stdout);
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
