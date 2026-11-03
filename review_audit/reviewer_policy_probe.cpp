// Small/large-number policy construction only; no huge parallel kernels.
#include <Kokkos_Core.hpp>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

using I = std::int32_t;
using P = Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>, Kokkos::IndexType<I>>;
using Wide = Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>, Kokkos::IndexType<std::int64_t>>;

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string kind = argv[1];
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const I mx = std::numeric_limits<I>::max();
    if (kind == "normal") {
      P p({0,0}, {8,8}, {2,2});
      std::cout << "normal tiles=" << p.impl_num_tiles() << '\n';
      rc = p.impl_num_tiles() == 16 ? 0 : 3;
    } else if (kind == "pr-explicit") {
      P p({0,0}, {1,mx-1}, {1,1});
      auto reco=p.tile_size_recommended();
      std::cout << "pr-explicit tiles=" << p.impl_num_tiles() << " reco="<<reco[0]<<","<<reco[1]<<'\n';
      rc=(p.impl_num_tiles()==mx-1 && reco[0]==1 && reco[1]==mx-1)?0:4;
    } else if (kind == "review-noexplicit") {
      // Reviewer's new suggested case: true Host recommended tiles, no huge kernel.
      P p({0,0}, {1,mx-1});
      auto t=p.tile_size();
      std::cout<<"review-noexplicit tiles="<<p.impl_num_tiles()<<" tile="<<t[0]<<","<<t[1]<<'\n';
      rc=(p.impl_num_tiles()==1 && t[0]==1 && t[1]==mx-1)?0:5;
    } else if (kind == "actual-product-overlimit") {
      // Flat extent fits int64_t, explicit tile product exceeds INT_MAX.
      Wide p({0,0},{46341,46341},{46341,46341});
      std::cout<<"actual-product-overlimit unexpectedly accepted: "<<p.impl_num_tiles()<<'\n';
      rc=6;
    } else if (kind == "tile-too-wide-for-int") {
      // A legal int64_t tile dimension too large for m_prod_tile_dims (int).
      const std::int64_t n = static_cast<std::int64_t>(mx) + 5;
      Wide p({0,0},{1,n},{1,n});
      std::cout<<"tile-too-wide-for-int unexpectedly accepted: "<<p.impl_num_tiles()<<'\n';
      rc=7;
    } else if (kind == "actual-md-parallel-reduce-tile1" ||
               kind == "actual-md-parallel-reduce-tile4") {
      const int tile = kind == "actual-md-parallel-reduce-tile1" ? 1 : 4;
      P p({mx-1,0}, {mx,1}, {tile,1});
      std::atomic<int> calls{0}, invalid{0};
      std::int64_t result = 0;
      Kokkos::parallel_reduce("review-audit-execution", p,
          [&](I i, I j, std::int64_t& update) {
            calls.fetch_add(1);
            if (i != mx-1 || j != 0) invalid.fetch_add(1);
            update += 1;
          }, result);
      Kokkos::fence();
      std::cout << kind << " result=" << result << " calls=" << calls
                << " invalid=" << invalid << " expected=1\n";
      rc = result == 1 && calls == 1 && invalid == 0 ? 0 : 9;
    } else if (kind == "construct-bench") {
      constexpr int rounds = 100000;
      std::int64_t checksum = 0;
      const auto begin = std::chrono::steady_clock::now();
      for (int i=0;i<rounds;++i) {
        // Vary bounds so the optimizer cannot fold an invariant constructor.
        P p({0,0},{16+i%7,16},{4,4});
        checksum += p.impl_num_tiles();
      }
      const auto end = std::chrono::steady_clock::now();
      const auto us=std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
      std::cout << "construct-bench iterations=" << rounds << " us=" << us
                << " checksum=" << checksum << '\n';
      rc=checksum>0?0:10;
    } else if (kind == "division-boundary-contract") {
      // At equality, '<' is different from '<='. Reviewer proposed '<',
      // but current <= policy accepts tile dimension exactly equal to cap.
      constexpr std::int64_t cap=1024, product=512, next=2;
      const bool safe_original=(product*next<=cap);
      const bool equivalent=(product<=cap/next);
      const bool strict=(product<cap/next);
      std::cout<<"equality original="<<safe_original<<" equivalent="<<equivalent
               <<" strict="<<strict<<'\n';
      rc=(safe_original && equivalent && !strict)?0:8;
    } else {
      rc=2;
    }
  }
  Kokkos::finalize();
  return rc;
}
