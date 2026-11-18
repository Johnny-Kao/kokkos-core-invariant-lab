#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <numeric>

#ifdef KOKKOS_ENABLE_OPENMP
using Exec = Kokkos::OpenMP;
#else
using Exec = Kokkos::Serial;
#endif

struct Matrix {
  // Identity default: necessary for the scan identity in TeamThreadRange.
  int a = 1, b = 0, c = 0, d = 1;
  KOKKOS_FUNCTION Matrix() = default;
  KOKKOS_FUNCTION Matrix(int a_, int b_, int c_, int d_)
      : a(a_), b(b_), c(c_), d(d_) {}
};

struct MatMul {
  KOKKOS_INLINE_FUNCTION
  Matrix operator()(const Matrix& x, const Matrix& y) const {
    return Matrix{x.a * y.a + x.b * y.c,
                  x.a * y.b + x.b * y.d,
                  x.c * y.a + x.d * y.c,
                  x.c * y.b + x.d * y.d};
  }
};

struct Identity {
  KOKKOS_INLINE_FUNCTION Matrix operator()(Matrix x) const { return x; }
};

static bool equal(const Matrix& x, const Matrix& y) {
  return x.a == y.a && x.b == y.b && x.c == y.c && x.d == y.d;
}
static void print_matrix(const char* tag, Matrix x) {
  std::printf("%s=[%d,%d;%d,%d]\n", tag, x.a, x.b, x.c, x.d);
}

int main(int argc, char** argv) {
  if (argc != 2 ||
      (std::strcmp(argv[1], "--expect-red") != 0 &&
       std::strcmp(argv[1], "--expect-green") != 0)) {
    std::puts("usage: k04_scan_probe --expect-red|--expect-green");
    return 2;
  }
  const bool red = std::strcmp(argv[1], "--expect-red") == 0;
  Kokkos::initialize();

  std::array<Matrix, 3> input_data{
      Matrix{1, 1, 0, 1}, Matrix{1, 0, 1, 1}, Matrix{2, 0, 0, 1}};
  const Matrix init{1, 0, 1, 1};
  std::array<Matrix, 3> oracle, oracle_no_init;
  std::transform_inclusive_scan(
      input_data.begin(), input_data.end(), oracle.begin(), MatMul{},
      Identity{}, init);
  std::transform_inclusive_scan(
      input_data.begin(), input_data.end(), oracle_no_init.begin(), MatMul{},
      Identity{});

  Kokkos::View<Matrix*, Exec> in("k04-matrices", 3);
  Kokkos::View<Matrix*, Exec> exec_out("k04-exec-out", 3);
  Kokkos::View<Matrix*, Exec> exec_no_init("k04-exec-noinit", 3);
  Kokkos::View<Matrix*, Exec> team_out("k04-team-out", 3);
  auto host_input = Kokkos::create_mirror_view(in);
  for (int i = 0; i < 3; ++i) host_input(i) = input_data[i];
  Kokkos::deep_copy(in, host_input);

  namespace KE = Kokkos::Experimental;
  KE::transform_inclusive_scan(Exec{}, in, exec_out, MatMul{}, Identity{}, init);
  KE::transform_inclusive_scan(Exec{}, in, exec_no_init, MatMul{}, Identity{});

  using Team = typename Kokkos::TeamPolicy<Exec>::member_type;
  Kokkos::parallel_for(
      "k04-valid-TeamPolicy-scan", Kokkos::TeamPolicy<Exec>(1, 1),
      KOKKOS_LAMBDA(const Team& member) {
        KE::transform_inclusive_scan(member, in, team_out, MatMul{}, Identity{},
                                     init);
      });
  Kokkos::fence();

  const auto exec_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, exec_out);
  const auto no_init_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, exec_no_init);
  const auto team_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, team_out);

  bool control_ok = true, exec_correct = true, team_correct = true;
  for (int i = 0; i < 3; ++i) {
    control_ok = control_ok && equal(no_init_host(i), oracle_no_init[i]);
    exec_correct = exec_correct && equal(exec_host(i), oracle[i]);
    team_correct = team_correct && equal(team_host(i), oracle[i]);
  }
  print_matrix("oracle_first", oracle[0]);
  print_matrix("exec_first", exec_host(0));
  print_matrix("team_first", team_host(0));
  std::printf("K04_RESULTS control=%d exec_correct=%d team_correct=%d\n",
              int(control_ok), int(exec_correct), int(team_correct));
  Kokkos::finalize();

  if (!control_ok) {
    std::puts("K04_CONTROL_FAILURE: without-init API differs from standard");
    return 3;
  }
  if (red && !exec_correct && !team_correct) {
    std::puts("K04_PUBLIC_API_RED: wrong matrix-prefix products, no error");
    return 0;
  }
  if (!red && exec_correct && team_correct) {
    std::puts("K04_PUBLIC_API_GREEN: matches standard matrix-prefix products");
    return 0;
  }
  std::puts("K04_EXPECTATION_MISMATCH");
  return 4;
}
