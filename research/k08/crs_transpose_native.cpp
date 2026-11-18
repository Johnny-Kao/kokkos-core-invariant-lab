// K08 independent research test: uses ORIGINAL Kokkos 5.3.0 public Crs API.
// Source stays in private lab, no upstream patch or issue.
// Square control must pass; rectangular case probes dimension assumption.
// Requires actual Kokkos::Serial backend with ASan/UBSan enabled.
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const bool rectangular = std::strcmp(argv[1],"rectangular")==0;
  const bool square = std::strcmp(argv[1],"square")==0;
  if (!rectangular && !square) return 2;
  Kokkos::initialize(argc, argv);
  {
    using Space = Kokkos::Serial;
    using Graph = Kokkos::Crs<int,Space>;
    using SizeType = Graph::size_type;
    Graph in;
    // Both graphs have 2 rows with 3 entries; the 'rectangular'
    // graph is a legitimate 2 x 4 CRS if rectangular transpose is supported.
    in.row_map=Graph::row_map_type("input_row_map",3);
    in.entries=Graph::entries_type("input_entries",3);
    auto hrows=Kokkos::create_mirror_view(in.row_map);
    hrows(0)=static_cast<SizeType>(0);
    hrows(1)=static_cast<SizeType>(2);
    hrows(2)=static_cast<SizeType>(3);
    auto hentries=Kokkos::create_mirror_view(in.entries);
    hentries(0)=0;
    hentries(1)=rectangular ? 3 : 1;
    hentries(2)=1;
    Kokkos::deep_copy(in.row_map,hrows);
    Kokkos::deep_copy(in.entries,hentries);
    std::printf("K08_NATIVE_INPUT rows=2 implied_columns=%d max_index=%d\n", rectangular?4:2, rectangular?3:1);
    std::fflush(stdout);
    Graph transposed;
    Kokkos::transpose_crs(transposed,in);
    Kokkos::fence("K08 native CRS transpose");
    std::printf("K08_NATIVE_COMPLETED input_rows=%lld transpose_rows=%lld\n",
      static_cast<long long>(in.numRows()),
      static_cast<long long>(transposed.numRows()));
  }
  Kokkos::finalize();
  return 0;
}
