cmake_minimum_required(VERSION 3.25)
project(kokkos_mdrange_recommendation_check LANGUAGES CXX)
find_package(Kokkos CONFIG REQUIRED)
add_executable(tile_recommendation_probe probe.cpp)
target_link_libraries(tile_recommendation_probe PRIVATE Kokkos::kokkos)
target_compile_features(tile_recommendation_probe PRIVATE cxx_std_20)
target_compile_options(tile_recommendation_probe PRIVATE
  -fsanitize=signed-integer-overflow
  -fno-sanitize-recover=signed-integer-overflow
  -fno-omit-frame-pointer -g)
target_link_options(tile_recommendation_probe PRIVATE -fsanitize=signed-integer-overflow)
