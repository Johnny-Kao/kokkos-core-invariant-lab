#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>

using Exec = Kokkos::Serial;
using Key = std::int64_t;
using Keys = Kokkos::View<Key*, Exec>;

static constexpr Key base = INT64_C(9007199254740992); // 2^53

bool check_keys(const Keys& v, Key a, Key b) {
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
  return host(0) == -777 && host(1) == a && host(2) == b &&
         host(3) == 999;
}

Keys make_keys(Key larger, Key smaller) {
  Keys v("k03-keys", 4);
  auto host = Kokkos::create_mirror_view(v);
  host(0) = -777;
  host(1) = larger;
  host(2) = smaller;
  host(3) = 999;
  Kokkos::deep_copy(v, host);
  return v;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::puts("Usage: k03_sort_probe --controls|--bin-diagnostic|--subrange");
    return 2;
  }
  Kokkos::initialize();
  int result = 0;
  const char* mode = argv[1];

  if (!std::strcmp(mode, "--controls")) {
    Keys small = make_keys(7, 6);
    Kokkos::sort(Exec{}, small, std::size_t{1}, std::size_t{3});
    if (!check_keys(small, 6, 7)) {
      std::puts("K03_CONTROL_FAILED: ordinary int64 subrange");
      result = 10;
    }

    // Full-range Kokkos::sort takes the std::sort path on Serial.
    Keys full("k03-full", 2);
    auto host = Kokkos::create_mirror_view(full);
    host(0) = base + 1;
    host(1) = base;
    Kokkos::deep_copy(full, host);
    Kokkos::sort(Exec{}, full);
    auto after = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, full);
    if (after(0) != base || after(1) != base + 1) {
      std::puts("K03_CONTROL_FAILED: full-range int64 sort");
      result = 11;
    }

    if (!result) std::puts("K03_CONTROLS_PASSED");
  } else if (!std::strcmp(mode, "--bin-diagnostic")) {
    Keys keys = make_keys(base + 1, base);
    // This is the exact public BinOp1D used by the subrange path;
    // double(base) == double(base + 1) even though the integers differ.
    Kokkos::BinOp1D<Keys> op(1, base, base + 1);
    std::printf("K03_BIN_PRECISION: min=%lld max=%lld double_diff=%.0f\n",
                static_cast<long long>(base),
                static_cast<long long>(base + 1),
                static_cast<double>(base + 1) -
                    static_cast<double>(base));
    const int index = op.bin(keys, INT64_C(1));
    std::printf("K03_BIN_RESULT: index=%d, max_bins=%d\n", index,
                op.max_bins());
    if (index < 0 || index >= op.max_bins()) result = 20;
  } else if (!std::strcmp(mode, "--subrange")) {
    Keys keys = make_keys(base + 1, base);
    Kokkos::sort(Exec{}, keys, std::size_t{1}, std::size_t{3});
    if (!check_keys(keys, base, base + 1)) {
      std::puts("K03_WRONG_SORTED_RESULT");
      result = 21;
    } else {
      std::puts("K03_SORT_RESULT_CORRECT");
    }
  } else {
    result = 2;
  }

  Kokkos::finalize();
  return result;
}
