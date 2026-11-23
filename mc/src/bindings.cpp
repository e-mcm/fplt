#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "fplt_mc/build_info.hpp"
#include "fplt_mc/mathx.hpp"
#include "fplt_mc/points.hpp"
#include "fplt_mc/rng.hpp"
#include "fplt_mc/rules.hpp"
#include "fplt_mc/simulate.hpp"

namespace nb = nanobind;

namespace {

using F64_1 = nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig>;
using F64_2 = nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig>;
using I64_1 = nb::ndarray<nb::numpy, std::int64_t, nb::ndim<1>, nb::c_contig>;
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

fplt_mc::MatchOtherTables tables_from_arrays(F64_2 delta_goals, F64_2 delta_cs,
                                             F64_2 delta_scored) {
  if (delta_goals.shape(0) != fplt_mc::kNPositions || delta_goals.shape(1) != 3) {
    throw std::invalid_argument("delta_goals must have shape (4, 3)");
  }
  if (delta_cs.shape(0) != fplt_mc::kNPositions || delta_cs.shape(1) != 2) {
    throw std::invalid_argument("delta_cs must have shape (4, 2)");
  }
  if (delta_scored.shape(0) != fplt_mc::kNPositions || delta_scored.shape(1) != 2) {
    throw std::invalid_argument("delta_scored must have shape (4, 2)");
  }
  fplt_mc::MatchOtherTables tables;
  for (std::size_t p = 0; p < fplt_mc::kNPositions; ++p) {
    for (std::size_t b = 0; b < 3; ++b) {
      tables.delta_goals[p][b] = delta_goals.data()[p * 3 + b];
    }
    for (std::size_t b = 0; b < 2; ++b) {
      tables.delta_cs[p][b] = delta_cs.data()[p * 2 + b];
      tables.delta_scored[p][b] = delta_scored.data()[p * 2 + b];
    }
  }
  return tables;
}

nb::dict simulate_fixture_py(
    I64_1 home_goals, I64_1 away_goals, U8_1 side, U8_1 position, F64_2 minutes,
    F64_1 goal_weight, F64_1 assist_weight, F64_2 other_points,
    std::int32_t appearance_long_minutes, double appearance_short,
    double appearance_long, F64_1 goals_by_pos, double assists_pts, F64_1 cs_by_pos,
    std::int32_t conceded_every, F64_1 conceded_by_pos, double own_goal_weight,
    double p_assist, double cs_red_void_prob, bool conceded_scaled, bool sample_other,
    std::uint64_t kernel_seed, nb::object delta_goals_obj, nb::object delta_cs_obj,
    nb::object delta_scored_obj, bool return_samples) {
  const std::size_t players = minutes.shape(0);
  const std::size_t n = minutes.shape(1);
  if (home_goals.shape(0) != n || away_goals.shape(0) != n) {
    throw std::invalid_argument("home/away goals length must equal minutes.shape[1]");
  }
  if (side.shape(0) != players || position.shape(0) != players ||
      goal_weight.shape(0) != players || assist_weight.shape(0) != players) {
    throw std::invalid_argument("player vectors must match minutes.shape[0]");
  }
  if (other_points.shape(0) != players || other_points.shape(1) != 3) {
    throw std::invalid_argument("other_points must have shape (players, 3)");
  }
  if (!(p_assist >= 0.0 && p_assist <= 1.0) || own_goal_weight < 0.0 ||
      !(cs_red_void_prob >= 0.0 && cs_red_void_prob <= 1.0)) {
    throw std::invalid_argument("invalid sim params");
  }

  const fplt_mc::RulesPOD rules = rules_from_arrays(
      appearance_long_minutes, appearance_short, appearance_long, goals_by_pos,
      assists_pts, cs_by_pos, conceded_every, conceded_by_pos);

  fplt_mc::SimParams params;
  params.own_goal_weight = own_goal_weight;
  params.p_assist = p_assist;
  params.cs_red_void_prob = cs_red_void_prob;
  params.conceded_scaled = conceded_scaled;
  params.sample_other = sample_other;

  const bool has_match = !delta_goals_obj.is_none();
  if (has_match != !delta_cs_obj.is_none() ||
      has_match != !delta_scored_obj.is_none()) {
    throw std::invalid_argument(
        "delta_goals, delta_cs, and delta_scored must all be set or all None");
  }
  fplt_mc::MatchOtherTables tables{};
  const fplt_mc::MatchOtherTables* match_ptr = nullptr;
  if (has_match) {
    tables = tables_from_arrays(nb::cast<F64_2>(delta_goals_obj),
                                nb::cast<F64_2>(delta_cs_obj),
                                nb::cast<F64_2>(delta_scored_obj));
    match_ptr = &tables;
  }

  std::vector<std::int64_t> goals(players * n);
  std::vector<std::int64_t> assists(players * n);
  std::vector<double> points(players * n);
  std::vector<std::uint8_t> clean_sheets(players * n);
  std::vector<std::int64_t> unattr_home(n);
  std::vector<std::int64_t> unattr_away(n);
  fplt_mc::MatchSamples samples;
  samples.goals = goals.data();
  samples.assists = assists.data();
  samples.points = points.data();
  samples.clean_sheets = clean_sheets.data();
  samples.unattributed_home = unattr_home.data();
  samples.unattributed_away = unattr_away.data();

  std::vector<double> xpts(players);
  std::vector<double> xpts_sd(players);
  std::vector<double> p_played(players);
  std::vector<double> p_60(players);
  std::vector<double> p_goal(players);
  std::vector<double> p_assist_arr(players);
  std::vector<double> p_cs(players);
  std::vector<double> p_haul(players);
  fplt_mc::FixtureSummary summary;
  summary.xpts = xpts.data();
  summary.xpts_sd = xpts_sd.data();
  summary.p_played = p_played.data();
  summary.p_60 = p_60.data();
  summary.p_goal = p_goal.data();
  summary.p_assist = p_assist_arr.data();
  summary.p_cs = p_cs.data();
  summary.p_haul = p_haul.data();

  fplt_mc::simulate_fixture(home_goals.data(), away_goals.data(), side.data(),
                            position.data(), minutes.data(), goal_weight.data(),
                            assist_weight.data(), other_points.data(), rules, params,
                            match_ptr, kernel_seed, players, n, &samples, &summary);

  nb::dict out;
  out["xpts"] = owned_array(std::move(xpts), {players});
  out["xpts_sd"] = owned_array(std::move(xpts_sd), {players});
  out["p_played"] = owned_array(std::move(p_played), {players});
  out["p_60"] = owned_array(std::move(p_60), {players});
  out["p_goal"] = owned_array(std::move(p_goal), {players});
  out["p_assist"] = owned_array(std::move(p_assist_arr), {players});
  out["p_cs"] = owned_array(std::move(p_cs), {players});
  out["p_haul"] = owned_array(std::move(p_haul), {players});
  out["unattributed_home"] = owned_array(std::move(unattr_home), {n});
  out["unattributed_away"] = owned_array(std::move(unattr_away), {n});
  if (return_samples) {
    out["goals"] = owned_array(std::move(goals), {players, n});
    out["assists"] = owned_array(std::move(assists), {players, n});
    out["points"] = owned_array(std::move(points), {players, n});
    out["clean_sheets"] = owned_array(std::move(clean_sheets), {players, n});
  }
  return out;
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "fplt-mc native Monte Carlo kernel (production simulate_fixture + Philox).";
  m.attr("KERNEL_API") = fplt_mc::KERNEL_API;
  m.def("build_info", &fplt_mc::build_info,
        "Return version, git SHA, compiler, and flags.");
  m.def(
      "philox_u01",
      [](std::uint64_t seed, std::uint32_t stream, std::uint32_t sim,
         std::uint32_t slot) { return fplt_mc::philox_u01(seed, stream, sim, slot); },
      nb::arg("seed"), nb::arg("stream"), nb::arg("sim"), nb::arg("slot"),
      "Draw one Philox uniform in [0, 1).");
  m.def(
      "mathx_exp", [](double x) { return fplt_mc::mathx::exp(x); }, nb::arg("x"),
      "Portable exp (CORE-MATH stub in PR1).");

  m.def("simulate_fixture", &simulate_fixture_py, nb::arg("home_goals"),
        nb::arg("away_goals"), nb::arg("side"), nb::arg("position"), nb::arg("minutes"),
        nb::arg("goal_weight"), nb::arg("assist_weight"), nb::arg("other_points"),
        nb::arg("appearance_long_minutes"), nb::arg("appearance_short"),
        nb::arg("appearance_long"), nb::arg("goals_by_pos"), nb::arg("assists_pts"),
        nb::arg("cs_by_pos"), nb::arg("conceded_every"), nb::arg("conceded_by_pos"),
        nb::arg("own_goal_weight"), nb::arg("p_assist"), nb::arg("cs_red_void_prob"),
        nb::arg("conceded_scaled"), nb::arg("sample_other"), nb::arg("kernel_seed"),
        nb::arg("delta_goals") = nb::none(), nb::arg("delta_cs") = nb::none(),
        nb::arg("delta_scored") = nb::none(), nb::arg("return_samples") = false,
        "Allocate + points + aggregate using Philox streams from kernel_seed.");
}
