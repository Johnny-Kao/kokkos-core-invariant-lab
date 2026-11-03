// Public API production-path test: actual Kokkos parallel_for / parallel_reduce.
// Counters and bounds checks prevent dangerous out-of-range writes.
#include <Kokkos_Core.hpp>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(KOKKOS_ENABLE_OPENMP)
using Exec = Kokkos::OpenMP;
#elif defined(KOKKOS_ENABLE_THREADS)
using Exec = Kokkos::Threads;
#else
#error "This test requires an enabled multithreaded host backend"
#endif

using I = std::int32_t;
using RP = Kokkos::RangePolicy<Exec, Kokkos::IndexType<I>>;
using MP = Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>, Kokkos::IndexType<I>>;

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  Kokkos::initialize(Kokkos::InitializationSettings().set_num_threads(4));
  int status = 0;
  {
    const std::string mode(argv[1]);
    const I hi = std::numeric_limits<I>::max();
    std::atomic<std::int64_t> calls{0}, out{0};
    if (mode == "baseline-range") {
      std::vector<std::atomic<int>> hits(1024);
      for (auto& v : hits) v.store(0);
      RP p(I(0), I(1024), Kokkos::ChunkSize(1));
      Kokkos::parallel_for("baseline-range", p, [&](I i) {
        calls.fetch_add(1);
        if (i < 0 || i >= 1024) out.fetch_add(1);
        else hits[static_cast<std::size_t>(i)].fetch_add(1);
      });
      Kokkos::fence();
      int bad = 0;
      for (auto& v : hits) if (v.load() != 1) ++bad;
      std::cout << "baseline-range calls=" << calls << " out=" << out << " wrong_counts=" << bad << "\\n";
      if (calls != 1024 || out || bad) status = 10;
    } else if (mode == "boundary-range") {
      RP p(hi-1, hi, Kokkos::ChunkSize(1));
      Kokkos::parallel_for("boundary-range", p, [&](I i) {
        calls.fetch_add(1);
        if (i != hi-1) out.fetch_add(1);
      });
      Kokkos::fence();
      std::cout << "boundary-range calls=" << calls << " out=" << out << " expected=1\\n";
      if (calls != 1 || out) status = 11;
    } else if (mode == "baseline-range-reduce" || mode == "boundary-range-reduce") {
      const bool boundary = mode == "boundary-range-reduce";
      const I begin = boundary ? hi-1 : 0;
      const I end = boundary ? hi : 1024;
      const std::int64_t expected = boundary ? 1 : 1024;
      RP p(begin, end, Kokkos::ChunkSize(1));
      std::int64_t total = 0;
      Kokkos::parallel_reduce(mode, p, [&](I i, std::int64_t& acc) {
        acc += 1;
        calls.fetch_add(1);
        if (i < begin || i >= end) out.fetch_add(1);
      }, total);
      Kokkos::fence();
      std::cout << mode << " reduction=" << total << " calls=" << calls << " out=" << out << " expected=" << expected << "\\n";
      if (total != expected || calls != expected || out) status = 14;
    } else if (mode == "boundary-md-for" || mode == "boundary-md-reduce") {
      MP p({hi-1, 0}, {hi, 1}, {4, 1});
      if (mode == "boundary-md-for") {
        Kokkos::parallel_for("boundary-md-for", p, [&](I i, I j) {
          calls.fetch_add(1);
          if (i != hi-1 || j != 0) out.fetch_add(1);
        });
        Kokkos::fence();
      } else {
        std::int64_t total = 0;
        Kokkos::parallel_reduce("boundary-md-reduce", p, [&](I i, I j, std::int64_t& sum) {
          sum += 1;
          calls.fetch_add(1);
          if (i != hi-1 || j != 0) out.fetch_add(1);
        }, total);
        Kokkos::fence();
        std::cout << "reduction=" << total << " expected=1\\n";
        if (total != 1) status = 12;
      }
      std::cout << mode << " calls=" << calls << " out=" << out << " expected=1\\n";
      if (calls != 1 || out) status = 13;
    } else {
      status = 2;
    }
  }
  Kokkos::finalize();
  return status;
}
