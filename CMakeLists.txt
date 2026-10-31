cmake_minimum_required(VERSION 3.25)
project(kokkos_mdrange_invariant LANGUAGES CXX)
find_package(Kokkos CONFIG REQUIRED)
add_executable(mdrange_invariant_probe mdrange_invariant_probe.cpp)
target_link_libraries(mdrange_invariant_probe PRIVATE Kokkos::kokkos)
target_compile_features(mdrange_invariant_probe PRIVATE cxx_std_20)
target_compile_options(mdrange_invariant_probe PRIVATE
  -fsanitize=signed-integer-overflow -fno-sanitize-recover=signed-integer-overflow
  -fno-omit-frame-pointer -g)
target_link_options(mdrange_invariant_probe PRIVATE -fsanitize=signed-integer-overflow)
