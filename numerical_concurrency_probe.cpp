// Scientific-result and concurrency diagnostics using public host Kokkos APIs.
// All recordings are bounds-safe: no user arrays are indexed by faulty coordinates.
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(KOKKOS_ENABLE_OPENMP)
using Exec = Kokkos::OpenMP;
#elif defined(KOKKOS_ENABLE_THREADS)
using Exec = Kokkos::Threads;
#else
#error "Enable either OpenMP or Threads"
#endif

using I = std::int32_t;
using RP = Kokkos::RangePolicy<Exec, Kokkos::IndexType<I>>;
using MP = Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>, Kokkos::IndexType<I>>;

struct SafeRecorder {
  std::mutex mu;
  std::map<std::int64_t, std::map<std::thread::id, int>> hits;
  std::int64_t calls = 0;
  std::int64_t out = 0;
  std::int64_t lo, hi;

  SafeRecorder(std::int64_t a, std::int64_t b) : lo(a), hi(b) {}

  void record(I i) {
    const std::int64_t ix = static_cast<std::int64_t>(i);
    std::lock_guard<std::mutex> lock(mu);
    ++calls;
    if (ix < lo || ix >= hi) ++out;
    // One loop iteration is always small for these cases. Avoid unbounded logs.
    if (calls <= 256) ++hits[ix][std::this_thread::get_id()];
  }

  void report(const std::string& label, std::int64_t expected_calls,
              double observed_reduction, double expected_reduction) {
    std::set<std::thread::id> all_threads;
    int duplicate_valid = 0, overlap = 0, missed = 0, observed_valid = 0;
    for (auto& [ix, bythread] : hits) {
      int n = 0;
      for (auto& [tid, count] : bythread) {
        n += count;
        all_threads.insert(tid);
      }
      if (ix >= lo && ix < hi) {
        ++observed_valid;
        if (n > 1) ++duplicate_valid;
      }
      if (bythread.size() > 1) ++overlap;
    }
    const std::int64_t range = hi - lo;
    if (range < 256) missed = static_cast<int>(range) - observed_valid;
    std::cout << label
      << " expected_calls=" << expected_calls
      << " actual_calls=" << calls
      << " out=" << out
      << " distinct_workers=" << all_threads.size()
      << " duplicate_valid_indices=" << duplicate_valid
      << " cross_thread_same_index=" << overlap
      << " missing_valid_indices=" << missed
      << " expected_sum=" << expected_reduction
      << " actual_sum=" << observed_reduction
      << " abs_error=" << std::abs(observed_reduction - expected_reduction)
      << "\n";
    std::cout << "sample_indices:";
    for (auto& [ix, bythread] : hits) {
      int n = 0;
      for (auto& [id, count] : bythread) n += count;
      std::cout << " " << ix << ":" << n << ":" << bythread.size();
    }
    std::cout << "\n";
  }
};

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: probe [range|mdrange] [baseline|boundary] threads tile\n";
    return 2;
  }
  const std::string family = argv[1];
  const std::string boundary = argv[2];
  const int threads = std::stoi(argv[3]);
  const int tile = std::stoi(argv[4]);
  if ((family != "range" && family != "mdrange") ||
      (boundary != "baseline" && boundary != "boundary") ||
      threads < 1 || threads > 8 || tile < 1 || tile > 8) return 2;
  Kokkos::initialize(Kokkos::InitializationSettings().set_num_threads(threads));
  int rc = 0;
  {
    const bool high = boundary == "boundary";
    const I max = std::numeric_limits<I>::max();
    // Large baseline has a real 32-element workload; high-index case only one.
    const I first = high ? max - 1 : 0;
    const I last = high ? max : 32;
    const std::int64_t count = static_cast<std::int64_t>(last) - first;
    SafeRecorder record(first, last);
    double total = 0;
    // Each visited index contributes 1.25, including invalid calls.
    // This measures numerical correctness without out-of-range writes.
    if (family == "range") {
      RP policy(first, last, Kokkos::ChunkSize(1));
      Kokkos::parallel_reduce("range-result-audit", policy, [&](I i, double& sum) {
        record.record(i);
        sum += 1.25;
      }, total);
    } else {
      MP policy({first, 0}, {last, 1}, {tile, 1});
      Kokkos::parallel_reduce("mdrange-result-audit", policy, [&](I i, I j, double& sum) {
        record.record(i);
        if (j != 0) {
          // j should always be 0 under this test.
          // record it as an unexpected extra in a way that does not touch memory.
          record.record(-1);
        }
        sum += 1.25;
      }, total);
    }
    Kokkos::fence();
    const std::string label = family + "/" + boundary + "/threads=" +
                              std::to_string(threads) + "/tile=" + std::to_string(tile);
    record.report(label, count, total, count * 1.25);
    if (record.calls != count || record.out != 0 || total != count * 1.25) rc = 10;
  }
  Kokkos::finalize();
  return rc;
}
