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
#include "fplt_mc/points.hpp"
#include "fplt_mc/rng.hpp"
#include "fplt_mc/rules.hpp"

namespace nb = nanobind;

namespace {

using F64_1 = nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig>;
using F64_2 = nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig>;
using I64_1 = nb::ndarray<nb::numpy, std::int64_t, nb::ndim<1>, nb::c_contig>;
using I64_2 = nb::ndarray<nb::numpy, std::int64_t, nb::ndim<2>, nb::c_contig>;
using U8_1 = nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<1>, nb::c_contig>;

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

auto poisson_sequential_with_uniforms(F64_1 rate, F64_1 uniforms) {
  const std::size_t n = rate.shape(0);
  if (uniforms.shape(0) != n) {
    throw std::invalid_argument("uniforms length must equal rate length");
  }
  std::vector<std::int64_t> out(n);
  fplt_mc::poisson_sequential(rate.data(), uniforms.data(), n, out.data());
  return owned_array(std::move(out), {n});
}

fplt_mc::RulesPOD rules_from_arrays(std::int32_t appearance_long_minutes,
                                    double appearance_short, double appearance_long,
                                    F64_1 goals_by_pos, double assists_pts,
                                    F64_1 cs_by_pos, std::int32_t conceded_every,
                                    F64_1 conceded_by_pos) {
  if (goals_by_pos.shape(0) != fplt_mc::kNPositions ||
      cs_by_pos.shape(0) != fplt_mc::kNPositions ||
      conceded_by_pos.shape(0) != fplt_mc::kNPositions) {
    throw std::invalid_argument(
        "goals/cs/conceded position tables must have length 4 (GK,DEF,MID,FWD)");
  }
  fplt_mc::RulesPOD rules;
  rules.appearance_long_minutes = appearance_long_minutes;
  rules.appearance_short = appearance_short;
  rules.appearance_long = appearance_long;
  rules.assists = assists_pts;
  rules.goals_conceded_every = conceded_every;
  for (std::size_t p = 0; p < fplt_mc::kNPositions; ++p) {
    rules.goals[p] = goals_by_pos.data()[p];
    rules.clean_sheets[p] = cs_by_pos.data()[p];
    rules.goals_conceded_points[p] = conceded_by_pos.data()[p];
  }
  return rules;
}

nb::tuple points_with_uniforms(U8_1 position, U8_1 side, F64_2 minutes, I64_2 goals,
                               I64_2 assists, I64_1 home, I64_1 away,
                               F64_2 other_points, std::int32_t appearance_long_minutes,
                               double appearance_short, double appearance_long,
                               F64_1 goals_by_pos, double assists_pts, F64_1 cs_by_pos,
                               std::int32_t conceded_every, F64_1 conceded_by_pos,
                               double cs_red_void_prob, bool conceded_scaled,
                               bool sample_other, nb::object cs_void_u_obj,
                               nb::object other_u_obj) {
  const std::size_t players = minutes.shape(0);
  const std::size_t n = minutes.shape(1);
  if (position.shape(0) != players || side.shape(0) != players) {
    throw std::invalid_argument("position/side length must equal minutes.shape[0]");
  }
  if (goals.shape(0) != players || goals.shape(1) != n || assists.shape(0) != players ||
      assists.shape(1) != n) {
    throw std::invalid_argument("goals/assists must match minutes shape");
  }
  if (home.shape(0) != n || away.shape(0) != n) {
    throw std::invalid_argument("home/away length must equal minutes.shape[1]");
  }
  if (other_points.shape(0) != players || other_points.shape(1) != 3) {
    throw std::invalid_argument("other_points must have shape (players, 3)");
  }
  if (!(cs_red_void_prob >= 0.0 && cs_red_void_prob <= 1.0)) {
    throw std::invalid_argument("cs_red_void_prob must be in [0, 1]");
  }

  const fplt_mc::RulesPOD rules = rules_from_arrays(
      appearance_long_minutes, appearance_short, appearance_long, goals_by_pos,
      assists_pts, cs_by_pos, conceded_every, conceded_by_pos);

  const double* cs_void_ptr = nullptr;
  F64_2 cs_void_u;
  if (cs_red_void_prob > 0.0) {
    if (cs_void_u_obj.is_none()) {
      throw std::invalid_argument("cs_void_u required when cs_red_void_prob > 0");
    }
    cs_void_u = nb::cast<F64_2>(cs_void_u_obj);
    if (cs_void_u.shape(0) != players || cs_void_u.shape(1) != n) {
      throw std::invalid_argument("cs_void uniforms must match minutes shape");
    }
    cs_void_ptr = cs_void_u.data();
  }

  const double* other_ptr = nullptr;
  F64_2 other_u;
  if (sample_other) {
    if (other_u_obj.is_none()) {
      throw std::invalid_argument("other_u required when sample_other is true");
    }
    other_u = nb::cast<F64_2>(other_u_obj);
    if (other_u.shape(0) != players || other_u.shape(1) != n) {
      throw std::invalid_argument("other uniforms must match minutes shape");
    }
    other_ptr = other_u.data();
  }

  std::vector<double> points_out(players * n);
  std::vector<std::uint8_t> cs_out(players * n);
  fplt_mc::points(rules, position.data(), side.data(), minutes.data(), goals.data(),
                  assists.data(), home.data(), away.data(), other_points.data(),
                  cs_red_void_prob, conceded_scaled, sample_other, cs_void_ptr,
                  other_ptr, players, n, points_out.data(), cs_out.data());

  // Return clean_sheet as bool-compatible uint8; Python casts to bool.
  return nb::make_tuple(owned_array(std::move(points_out), {players, n}),
                        owned_array(std::move(cs_out), {players, n}));
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

  m.def("poisson_sequential_with_uniforms", &poisson_sequential_with_uniforms,
        nb::arg("rate"), nb::arg("uniforms"),
        "Sequential-search Poisson matching simulation._poisson_sequential.");

  m.def("points_with_uniforms", &points_with_uniforms, nb::arg("position"),
        nb::arg("side"), nb::arg("minutes"), nb::arg("goals"), nb::arg("assists"),
        nb::arg("home"), nb::arg("away"), nb::arg("other_points"),
        nb::arg("appearance_long_minutes"), nb::arg("appearance_short"),
        nb::arg("appearance_long"), nb::arg("goals_by_pos"), nb::arg("assists_pts"),
        nb::arg("cs_by_pos"), nb::arg("conceded_every"), nb::arg("conceded_by_pos"),
        nb::arg("cs_red_void_prob"), nb::arg("conceded_scaled"),
        nb::arg("sample_other"), nb::arg("cs_void_u") = nb::none(),
        nb::arg("other_u") = nb::none(),
        "Points + CS matching simulation._points with injected uniforms "
        "(match_other=None; K4 later).");
}
