#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#ifdef KOKKOS_ENABLE_OPENMP
using Exec = Kokkos::OpenMP;
#else
using Exec = Kokkos::Serial;
#endif

using Member = Kokkos::TeamPolicy<Exec>::member_type;
using Scratch = typename Exec::scratch_memory_space;
using IntScratchView = Kokkos::View<int*, Scratch, Kokkos::MemoryUnmanaged>;

struct alignas(64) OverAlignedElement {
  unsigned char bytes[64];
};
using AlignedScratchView =
    Kokkos::View<OverAlignedElement*, Scratch, Kokkos::MemoryUnmanaged>;

struct Observation {
  int first = 0;
  int second = 0;
  int third = 0;
  int beyond = 0;
  int value = 0;
};

Observation run_case(std::size_t reserved_bytes, int mode) {
  Kokkos::View<int*, Exec> results("k02-team-outcomes", 5);
  Kokkos::deep_copy(results, -1);

  auto policy = Kokkos::TeamPolicy<Exec>(1, 1).set_scratch_size(
      0, Kokkos::PerTeam(reserved_bytes));

  Kokkos::parallel_for(
      "k02-real-TeamPolicy", policy,
      KOKKOS_LAMBDA(const Member& team) {
        if (mode == 0) {
          // Correctly budgeted two Views, as in Kokkos examples.
          IntScratchView first(team.team_shmem(), 1);
          IntScratchView second(team.team_shmem(), 1);
          results(0) = first.data() != nullptr;
          results(1) = second.data() != nullptr;
          if (first.data() && second.data()) {
            first(0) = 23;
            second(0) = 19;
            results(4) = first(0) + second(0);
          }
        } else if (mode == 1) {
          // Common 8-byte alignment: third View must be rejected when
          // ordinary scratch capacity is exhausted.
          IntScratchView first(team.team_shmem(), 1);
          IntScratchView second(team.team_shmem(), 1);
          IntScratchView third(team.team_shmem(), 1);
          results(0) = first.data() != nullptr;
          results(1) = second.data() != nullptr;
          results(2) = third.data() != nullptr;
        } else {
          // Real TeamPolicy scratch API, but intentionally under-budgeted.
          // Consume all declared bytes, then request an over-aligned View.
          // Never dereference that View: only inspect the returned address.
          void* base = team.team_shmem().get_shmem(reserved_bytes);
          AlignedScratchView next(team.team_shmem(), 1);
          results(0) = base != nullptr;
          results(2) = next.data() != nullptr;
          if (base && next.data()) {
            const auto base_addr = reinterpret_cast<std::uintptr_t>(base);
            const auto next_addr =
                reinterpret_cast<std::uintptr_t>(next.data());
            results(3) = next_addr >= base_addr + reserved_bytes;
            results(4) = static_cast<int>(next_addr - base_addr);
          }
        }
      });
  Kokkos::fence();
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
  Observation observation;
  observation.first = host(0);
  observation.second = host(1);
  observation.third = host(2);
  observation.beyond = host(3);
  observation.value = host(4);
  return observation;
}

int main(int argc, char** argv) {
  if (argc != 2 || (std::strcmp(argv[1], "--expect-red") != 0 &&
                    std::strcmp(argv[1], "--expect-green") != 0)) {
    std::puts("Usage: team_policy_probe --expect-red|--expect-green");
    return 2;
  }
  const bool expect_red = std::strcmp(argv[1], "--expect-red") == 0;
  Kokkos::initialize();
  const auto normal = run_case(2 * IntScratchView::shmem_size(1), 0);
  const auto small = run_case(16, 1);
  const auto aligned16 = run_case(16, 2);
  const auto aligned32 = run_case(32, 2);
  Kokkos::finalize();

  std::printf("proper_view_budget: first=%d second=%d sum=%d\n",
              normal.first, normal.second, normal.value);
  std::printf("ordinary_insufficient_budget: first=%d second=%d third=%d\n",
              small.first, small.second, small.third);
  std::printf("overaligned_after_16: first=%d pointer=%d beyond=%d offset=%d\n",
              aligned16.first, aligned16.third, aligned16.beyond,
              aligned16.value);
  std::printf("overaligned_after_32: first=%d pointer=%d beyond=%d offset=%d\n",
              aligned32.first, aligned32.third, aligned32.beyond,
              aligned32.value);

  if (normal.first != 1 || normal.second != 1 || normal.value != 42 ||
      small.first != 1 || small.second != 1 || small.third != 0 ||
      aligned16.first != 1 || aligned32.first != 1) {
    std::puts("K02_TEAM_CONTROL_FAILURE");
    return 3;
  }

  const bool returned_outside =
      aligned16.beyond == 1 || aligned32.beyond == 1;
  if (expect_red && returned_outside) {
    std::puts("K02_TEAM_REAL_PATH_RED: over-aligned View returned out-of-capacity pointer");
    return 0;
  }
  if (!expect_red && !returned_outside && aligned16.third == 0 &&
      aligned32.third == 0) {
    std::puts("K02_TEAM_REAL_PATH_GREEN: out-of-capacity View safely rejected");
    return 0;
  }
  std::puts("K02_TEAM_EXPECTATION_MISMATCH");
  return 4;
}
