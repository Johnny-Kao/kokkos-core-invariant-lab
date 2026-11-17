cmake_minimum_required(VERSION 3.22)
project(kokkos_scratch_k02 LANGUAGES CXX)

find_package(Kokkos CONFIG REQUIRED)

add_executable(k02_scratch_probe probe.cpp)
target_compile_features(k02_scratch_probe PRIVATE cxx_std_20)
target_link_libraries(k02_scratch_probe PRIVATE Kokkos::kokkos)

add_executable(k02_team_policy_probe team_policy_probe.cpp)
target_compile_features(k02_team_policy_probe PRIVATE cxx_std_20)
target_link_libraries(k02_team_policy_probe PRIVATE Kokkos::kokkos)
