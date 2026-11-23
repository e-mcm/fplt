#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "fplt_mc/allocate.hpp"
#include "fplt_mc/rng.hpp"

namespace nb = nanobind;

namespace {

using F64_1 = nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig>;
using F64_2 = nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig>;
using I64_1 = nb::ndarray<nb::numpy, std::int64_t, nb::ndim<1>, nb::c_contig>;
using I64_2 = nb::ndarray<nb::numpy, std::int64_t, nb::ndim<2>, nb::c_contig>;

template <typename T>
nb::ndarray<nb::numpy, T, nb::c_contig> owned_array(std::vector<T>&& data,
                                                    std::vector<size_t> shape) {
  T* ptr = data.data();
  auto* heap = new std::vector<T>(std::move(data));
  nb::capsule owner(heap,
                    [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return nb::ndarray<nb::numpy, T, nb::c_contig>(ptr, shape.size(), shape.data(),
                                                 owner);
}

auto pick_with_uniforms(F64_2 weights, F64_1 draw, double extra) {
  const std::size_t players = weights.shape(0);
  const std::size_t n = weights.shape(1);
  if (draw.shape(0) != n) {
    throw std::invalid_argument("draw length must equal weights.shape[1]");
  }
  std::vector<std::int64_t> out(n);
  fplt_mc::pick(weights.data(), draw.data(), extra, players, n, out.data());
  return owned_array(std::move(out), {n});
}

nb::tuple allocate_side_with_uniforms(I64_1 team_goals, F64_2 minutes,
                                      F64_1 goal_weight, F64_1 assist_weight,
                                      double own_goal_weight, double p_assist,
                                      F64_2 scorer_u, F64_2 assist_flag_u,
                                      F64_2 assister_u) {
  const std::size_t players = minutes.shape(0);
  const std::size_t n = minutes.shape(1);
  if (team_goals.shape(0) != n) {
    throw std::invalid_argument("team_goals length must equal minutes.shape[1]");
  }
  if (goal_weight.shape(0) != players || assist_weight.shape(0) != players) {
    throw std::invalid_argument(
        "goal_weight and assist_weight length must equal minutes.shape[0]");
  }
  if (own_goal_weight < 0.0) {
    throw std::invalid_argument("own_goal_weight must be non-negative");
  }
  if (!(p_assist >= 0.0 && p_assist <= 1.0)) {
    throw std::invalid_argument("p_assist must be in [0, 1]");
  }

  std::int64_t max_goals_i = 0;
  for (std::size_t j = 0; j < n; ++j) {
    max_goals_i = std::max(max_goals_i, team_goals.data()[j]);
  }
  if (max_goals_i < 0) {
    throw std::invalid_argument("team_goals must be non-negative");
  }
  const std::size_t max_goals = static_cast<std::size_t>(max_goals_i);

  auto check_grid = [&](F64_2 grid, const char* name) {
    if (grid.shape(0) < max_goals || grid.shape(1) != n) {
      throw std::invalid_argument(std::string(name) + " uniforms must have shape (>= " +
                                  std::to_string(max_goals) + ", " + std::to_string(n) +
                                  ")");
    }
  };
  check_grid(scorer_u, "scorer");
  check_grid(assist_flag_u, "assist_flag");
  check_grid(assister_u, "assister");

  std::vector<std::int64_t> scored(players * n);
  std::vector<std::int64_t> assisted(players * n);
  std::vector<std::int64_t> missed(n);

  fplt_mc::allocate_side(team_goals.data(), minutes.data(), goal_weight.data(),
                         assist_weight.data(), own_goal_weight, p_assist,
                         scorer_u.data(), assist_flag_u.data(), assister_u.data(),
                         players, n, max_goals, scored.data(), assisted.data(),
                         missed.data());

  return nb::make_tuple(owned_array(std::move(scored), {players, n}),
                        owned_array(std::move(assisted), {players, n}),
                        owned_array(std::move(missed), {n}));
}

}  // namespace

NB_MODULE(_testing, m) {
  m.doc() =
      "fplt-mc testing hooks: injected-uniform entry points for L2 parity "
      "and Philox helpers for L0/L3.";

  m.def(
      "philox_u01_batch",
      [](std::uint64_t seed, std::uint32_t stream, std::uint32_t sim, std::uint32_t n) {
        std::vector<double> out(n);
        for (std::uint32_t i = 0; i < n; ++i) {
          out[i] = fplt_mc::philox_u01(seed, stream, sim, i);
        }
        return out;
      },
      nb::arg("seed"), nb::arg("stream"), nb::arg("sim"), nb::arg("n"),
      "Return n Philox uniforms for a fixed (seed, stream, sim).");

  m.def("pick_with_uniforms", &pick_with_uniforms, nb::arg("weights"), nb::arg("draw"),
        nb::arg("extra"),
        "Inverse-CDF player pick matching simulation._pick (injected u).");

  m.def("allocate_side_with_uniforms", &allocate_side_with_uniforms,
        nb::arg("team_goals"), nb::arg("minutes"), nb::arg("goal_weight"),
        nb::arg("assist_weight"), nb::arg("own_goal_weight"), nb::arg("p_assist"),
        nb::arg("scorer_u"), nb::arg("assist_flag_u"), nb::arg("assister_u"),
        "Allocate goals/assists matching _allocate_side with injected "
        "uniforms.");
}
