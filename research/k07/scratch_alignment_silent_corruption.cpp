// Isolated research for K07 — run only on Kokkos 5.3.0; not an upstream patch.
// Exercise the original library's public ScratchMemorySpace API using a
// scratch region that is fully contained in a larger, ASan-valid allocation.
// This distinguishes exceeding scratch bounds from exceeding allocation bounds.
#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {
  alignas(64) std::array<unsigned char, 256> backing{};
  backing.fill(0x5A);
  constexpr std::size_t scratch_offset = 8;
  constexpr std::size_t scratch_bytes  = 8;
  constexpr std::size_t request_bytes  = 8;
  constexpr std::size_t requested_alignment = 64;
  // This scratch starts at an 8-byte-aligned pointer, matching Kokkos::ScratchMemorySpace::ALIGN.
  auto* scratch_begin = backing.data() + scratch_offset;
  auto* scratch_end   = scratch_begin + scratch_bytes;
  Kokkos::ScratchMemorySpace<Kokkos::Serial> small(scratch_begin, scratch_bytes);

  auto* obtained = static_cast<unsigned char*>(
      small.get_shmem_aligned(request_bytes, requested_alignment));

  const auto begin_i  = reinterpret_cast<std::uintptr_t>(scratch_begin);
  const auto end_i    = reinterpret_cast<std::uintptr_t>(scratch_end);
  const auto got_i    = reinterpret_cast<std::uintptr_t>(obtained);
  const bool outside  = obtained && (got_i < begin_i || got_i >= end_i ||
                                    request_bytes > end_i - got_i);

  // If a broken allocator hands back a pointer inside the backing array but
  // past the scratch budget, a write silently changes the guard region.
  if (outside && got_i >= reinterpret_cast<std::uintptr_t>(backing.data()) &&
      got_i < reinterpret_cast<std::uintptr_t>(backing.data() + backing.size())) {
    *obtained = 0xC3;
  }
  const bool guard_overwritten = backing[64] == 0xC3;
  std::printf("K07_BAD_SCRATCH original_capacity=%zu align=%zu request=%zu returned_offset=%td outside_scratch=%d sentinel_modified=%d\n",
              scratch_bytes, requested_alignment, request_bytes,
              obtained ? obtained - backing.data() : -1,
              int(outside), int(guard_overwritten));

  backing.fill(0x5A);
  Kokkos::ScratchMemorySpace<Kokkos::Serial> sufficient(scratch_begin, 80u);
  auto* valid = static_cast<unsigned char*>(
      sufficient.get_shmem_aligned(request_bytes, requested_alignment));
  const auto valid_i = reinterpret_cast<std::uintptr_t>(valid);
  const bool valid_inside = valid != nullptr &&
      valid_i >= begin_i && valid_i + request_bytes <= begin_i + 80u;
  std::printf("K07_CONTROL adequate_capacity=80 returned_offset=%td in_scratch=%d\n",
              valid ? valid - backing.data() : -1, int(valid_inside));

  if (outside && guard_overwritten && valid_inside) {
    std::puts("K07_NATIVE_SOURCE_REPRODUCED: silent overwrite past advertised scratch region");
    return 0;
  }
  std::puts("K07_NATIVE_SOURCE_NOT_REPRODUCED");
  return 1;
}
