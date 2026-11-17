cmake_minimum_required(VERSION 3.25)
project(kokkos_k03_sort LANGUAGES CXX)

find_package(Kokkos CONFIG REQUIRED)
add_executable(k03_sort_probe probe.cpp)
target_compile_features(k03_sort_probe PRIVATE cxx_std_20)
target_link_libraries(k03_sort_probe PRIVATE Kokkos::kokkos)
target_compile_options(k03_sort_probe PRIVATE
  -g -O1 -fno-omit-frame-pointer
  -fsanitize=float-cast-overflow
  -fno-sanitize-recover=float-cast-overflow)
target_link_options(k03_sort_probe PRIVATE -fsanitize=float-cast-overflow)
