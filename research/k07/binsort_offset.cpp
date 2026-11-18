// PRIVATE K07 independent source-first regression probe.
// All keys are within [0,3]; both key and value ranges have four elements.
// Public BinSort API accepts different key and value offsets, but its final
// write-back appears to use key range_begin instead of values_range_begin.
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using Exec = Kokkos::OpenMP;
using Vec = Kokkos::View<int*, Exec>;
using Op = Kokkos::BinOp1D<Vec>;
using BS = Kokkos::BinSort<Vec, Op>;

int main(int argc, char** argv) {
  if (argc != 2 ||
      (std::strcmp(argv[1],"aligned") != 0 &&
       std::strcmp(argv[1],"silent") != 0 &&
       std::strcmp(argv[1],"oob") != 0)) {
    std::fprintf(stderr,"usage: k07_binsort_offset aligned|silent|oob\n");
    return 2;
  }
  bool aligned = std::strcmp(argv[1],"aligned") == 0;
  bool oob = std::strcmp(argv[1],"oob") == 0;
  Kokkos::initialize();
  {
    Vec keys("keys_with_normal_nonzero_range", 8);
    Vec vals("values", oob ? 4 : 12);
    Kokkos::parallel_for("normal_keys", Kokkos::RangePolicy<Exec>(0,8),
                        KOKKOS_LAMBDA(int i) {
      keys(i) = (i < 4) ? 0 : ((i==4)?3:(i==5)?0:(i==6)?2:1);
    });
    auto val_h=Kokkos::create_mirror_view(vals);
    for (size_t i=0; i<vals.extent(0); ++i) val_h(i)=900+int(i);
    if (aligned) {
      val_h(4)=30; val_h(5)=0; val_h(6)=20; val_h(7)=10;
    } else {
      val_h(0)=30; val_h(1)=0; val_h(2)=20; val_h(3)=10;
    }
    Kokkos::deep_copy(vals,val_h);
    Exec{}.fence();
    Op op(3,0,3);
    BS sorter(keys,4,8,op,false);
    sorter.create_permute_vector();
    auto perm=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sorter.get_permute_vector());
    for (int i=0;i<4;++i) {
      constexpr int expected[4]={5,7,6,4};
      if (static_cast<int>(perm(i))!=expected[i]) {
        std::fprintf(stderr,"K07_BAD_PERMUTATION at %d got %lld\n",
          i,static_cast<long long>(perm(i)));
        return 3;
      }
    }
    std::printf("K07_MODE=%s KEY_RANGE=[4,8) VALUE_RANGE=[%d,%d) VALUES_LENGTH=%zu\n",
       argv[1], aligned?4:0, aligned?8:4, vals.extent(0));
    std::fflush(stdout);
    sorter.sort(Exec{},vals,aligned?4:0,aligned?8:4);
    Exec{}.fence();
    if (!oob) {
      auto got=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, vals);
      int dest=aligned?4:0;
      const int expected[4]={0,10,20,30};
      bool expected_range_correct=true;
      for(int i=0;i<4;++i)
        if(got(dest+i)!=expected[i]) expected_range_correct=false;
      bool unrelated_range_preserved=true;
      if(!aligned){
        for(int i=4;i<8;++i)
          if(got(i)!=900+i) unrelated_range_preserved=false;
      }
      std::printf("K07_OBSERVED expected_range_correct=%d unrelated_range_preserved=%d\n",
        int(expected_range_correct),int(unrelated_range_preserved));
      for(int i=0;i<12;++i)std::printf("%d%c",got(i),i==11?'\n':' ');
      if(aligned) {
        if(!expected_range_correct) return 4;
        std::puts("K07_ALIGNED_PASS");
      } else {
        if(expected_range_correct && unrelated_range_preserved) {
          std::puts("K07_SILENT_NOT_REPRODUCED"); return 5;
        }
        std::puts("K07_SILENT_WRONG_RANGE_OBSERVED"); // Process returned normally without ASan.
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
