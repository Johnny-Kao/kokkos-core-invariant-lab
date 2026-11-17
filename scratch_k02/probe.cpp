#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
  const bool boundary = argc > 1 && std::strcmp(argv[1], "--boundary") == 0;
  Kokkos::initialize();

  // Backing storage is larger than the declared scratch capacity, so the
  // diagnostic never dereferences memory outside the actual C++ object.
  alignas(64) unsigned char backing[32] = {};
  using Scratch = Kokkos::DefaultHostExecutionSpace::scratch_memory_space;
  int result = 0;

  {
    Scratch sufficient(backing, std::size_t{16});
    const void* first = sufficient.get_shmem(std::size_t{7});
    const void* aligned =
        sufficient.get_shmem_aligned(std::size_t{1}, std::ptrdiff_t{8});
    if (first != static_cast<void*>(backing) ||
        aligned != static_cast<void*>(backing + 8)) {
      std::puts("K02_CONTROL_FAILED: valid aligned allocation");
      result = 10;
    }
  }

  {
    Scratch exhausted(backing, std::size_t{7});
    const void* first = exhausted.get_shmem(std::size_t{7});
    const void* ordinary_overflow = exhausted.get_shmem(std::size_t{1});
    if (first != static_cast<void*>(backing) ||
        ordinary_overflow != nullptr) {
      std::puts("K02_CONTROL_FAILED: ordinary capacity rejection");
      result = 11;
    } else if (boundary && result == 0) {
      const void* aligned_overflow =
          exhausted.get_shmem_aligned(std::size_t{1}, std::ptrdiff_t{8});
      if (aligned_overflow == nullptr) {
        std::puts("K02_BOUNDARY_REJECTED: capacity guard held");
      } else {
        const auto base = reinterpret_cast<std::uintptr_t>(backing);
        const auto ptr = reinterpret_cast<std::uintptr_t>(aligned_overflow);
        if (ptr >= base + 7) {
          std::printf(
              "K02_OBSERVED_OUT_OF_CAPACITY: offset=%zu declared_capacity=7\n",
              static_cast<std::size_t>(ptr - base));
          result = 42;
        } else {
          std::puts("K02_UNEXPECTED_POINTER: review test assumptions");
          result = 12;
        }
      }
    }
  }

  if (!boundary && result == 0) {
    std::puts("K02_CONTROLS_PASSED");
  }
  Kokkos::finalize();
  return result;
}
