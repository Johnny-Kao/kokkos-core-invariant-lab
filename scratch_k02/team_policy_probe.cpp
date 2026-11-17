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
          // Determine whether the *actual* backend scratch region has been
          // exhausted. This matters because host pools retain their high-water
          // allocation and OpenMP preallocates a larger initial region.
          void* first = team.team_shmem().get_shmem(reserved_bytes);
          void* extra = team.team_shmem().get_shmem(std::size_t{1});
          results(0) = first != nullptr;
          results(2) = extra != nullptr;
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
  // Both explicit capacities exceed the OpenMP initial scratch reservation
  // with OMP_NUM_THREADS=2. Verify the effective bounds with ordinary
  // allocation controls before interpreting aligned View addresses.
  constexpr std::size_t capacity_a = 4112;
  constexpr std::size_t capacity_b = 4128;
  const auto control_a = run_case(capacity_a, 1);
  const auto aligned_a = run_case(capacity_a, 2);
  const auto control_b = run_case(capacity_b, 1);
  const auto aligned_b = run_case(capacity_b, 2);
  const auto normal = run_case(2 * IntScratchView::shmem_size(1), 0);
  Kokkos::finalize();

  std::printf("proper_view_budget: first=%d second=%d sum=%d\n",
              normal.first, normal.second, normal.value);
  std::printf("ordinary_exhausted_%zu: first=%d extra=%d\n",
              capacity_a, control_a.first, control_a.third);
  std::printf("ordinary_exhausted_%zu: first=%d extra=%d\n",
              capacity_b, control_b.first, control_b.third);
  std::printf("overaligned_after_%zu: first=%d pointer=%d beyond=%d offset=%d\n",
              capacity_a, aligned_a.first, aligned_a.third, aligned_a.beyond,
              aligned_a.value);
  std::printf("overaligned_after_%zu: first=%d pointer=%d beyond=%d offset=%d\n",
              capacity_b, aligned_b.first, aligned_b.third, aligned_b.beyond,
              aligned_b.value);

  if (normal.first != 1 || normal.second != 1 || normal.value != 42 ||
      control_a.first != 1 || control_a.third != 0 ||
      control_b.first != 1 || control_b.third != 0 ||
      aligned_a.first != 1 || aligned_b.first != 1) {
    std::puts("K02_TEAM_CONTROL_FAILURE: actual scratch capacity not established");
    return 3;
  }

  const bool returned_outside =
      aligned_a.beyond == 1 || aligned_b.beyond == 1;
  if (expect_red && returned_outside) {
    std::puts("K02_TEAM_REAL_PATH_RED: over-aligned View returned out-of-capacity pointer");
    return 0;
  }
  if (!expect_red && !returned_outside && aligned_a.third == 0 &&
      aligned_b.third == 0) {
    std::puts("K02_TEAM_REAL_PATH_GREEN: out-of-capacity View safely rejected");
    return 0;
  }
  std::puts("K02_TEAM_EXPECTATION_MISMATCH");
  return 4;
}
