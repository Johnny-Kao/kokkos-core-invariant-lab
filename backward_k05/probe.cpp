#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(KOKKOS_ENABLE_OPENMP)
using Exec = Kokkos::OpenMP;
#else
using Exec = Kokkos::Serial;
#endif

namespace KE = Kokkos::Experimental;

static int count_differences(const Kokkos::View<int*, Exec>& v,
                             const std::vector<int>& expected,
                             const char* scenario, int iteration) {
  const auto h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
  int mismatches = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (h(i) != expected[i]) {
      if (mismatches == 0) {
        std::printf(
            "K05_MISMATCH %s round=%d at=%zu expected=%d observed=%d\\n",
            scenario, iteration, i, expected[i], h(i));
      }
      ++mismatches;
    }
  }
  return mismatches;
}

int main(int argc, char** argv) {
  if (argc != 2 ||
      (std::strcmp(argv[1], "--expect-red") != 0 &&
       std::strcmp(argv[1], "--expect-clean") != 0)) {
    std::puts("usage: k05_backward_probe --expect-red|--expect-clean");
    return 2;
  }
  const bool red = std::strcmp(argv[1], "--expect-red") == 0;
  Kokkos::ScopeGuard kokkos_scope(argc, argv);
  const std::size_t n = 262145;
  std::vector<int> input(n);
  for (std::size_t i = 0; i < n; ++i)
    input[i] = static_cast<int>((i * 17 + 31) % 1000000007);

  auto expected_shift = input;
  std::copy_backward(expected_shift.begin(), expected_shift.end() - 1,
                     expected_shift.end());
  auto expected_move = input;
  std::move_backward(expected_move.begin(), expected_move.end() - 1,
                     expected_move.end());

  Kokkos::View<int*, Exec> in("k05-in", n), disjoint("k05-disjoint", n);
  auto h = Kokkos::create_mirror_view(in);
  for (std::size_t i = 0; i < n; ++i) h(i) = input[i];

  // Control: non-overlapping output, so all iterations are independent.
  Kokkos::deep_copy(in, h);
  const auto ret =
      KE::copy_backward(Exec{}, KE::begin(in), KE::end(in), KE::end(disjoint));
  if (ret != KE::begin(disjoint)) {
    std::puts("K05_CONTROL_INVALID_RETURN");
    return 3;
  }
  const int control_diff = count_differences(disjoint, input, "disjoint", -1);

  int copy_diff = 0, move_diff = 0;
  const int rounds = 12;
  for (int iteration = 0; iteration < rounds; ++iteration) {
    Kokkos::deep_copy(in, h);
    KE::copy_backward(Exec{}, KE::begin(in), KE::end(in) - 1, KE::end(in));
    copy_diff += count_differences(in, expected_shift, "copy_backward",
                                    iteration);

    Kokkos::deep_copy(in, h);
    KE::move_backward(Exec{}, KE::begin(in), KE::end(in) - 1, KE::end(in));
    move_diff += count_differences(in, expected_move, "move_backward",
                                    iteration);
  }

  std::printf("K05_RESULTS control_diff=%d copy_diff=%d move_diff=%d rounds=%d\\n",
              control_diff, copy_diff, move_diff, rounds);
  if (control_diff) return 3;
  if (red) {
    if (copy_diff || move_diff) {
      std::puts("K05_PUBLIC_API_RED: valid right-shift differs from C++ standard");
      return 0;
    }
    std::puts("K05_EXPECTED_RED_NOT_OBSERVED");
    return 4;
  }
  if (copy_diff == 0 && move_diff == 0) {
    std::puts("K05_PUBLIC_API_CLEAN: all tests matched C++ standard");
    return 0;
  }
  std::puts("K05_EXPECTED_CLEAN_NOT_OBSERVED");
  return 4;
}
