cmake_minimum_required(VERSION 3.25)
project(kokkos_tile_boundary_invariant LANGUAGES CXX)
find_package(Kokkos CONFIG REQUIRED)

add_executable(tile_boundary_probe ../tile_boundary_probe.cpp)
target_link_libraries(tile_boundary_probe PRIVATE Kokkos::kokkos)
target_compile_features(tile_boundary_probe PRIVATE cxx_std_20)
target_compile_options(tile_boundary_probe PRIVATE
  -fsanitize=signed-integer-overflow -fno-sanitize-recover=signed-integer-overflow
  -fno-omit-frame-pointer -g)
target_link_options(tile_boundary_probe PRIVATE -fsanitize=signed-integer-overflow)

add_executable(tile_boundary_probe_plain ../tile_boundary_probe.cpp)
target_link_libraries(tile_boundary_probe_plain PRIVATE Kokkos::kokkos)
target_compile_features(tile_boundary_probe_plain PRIVATE cxx_std_20)
target_compile_options(tile_boundary_probe_plain PRIVATE -O2)
