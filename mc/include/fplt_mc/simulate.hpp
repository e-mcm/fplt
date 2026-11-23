#pragma once

#include <cstddef>
#include <cstdint>

#include "fplt_mc/points.hpp"
#include "fplt_mc/rules.hpp"

namespace fplt_mc {

/// Per-player summary statistics (K5), matching ``composite._fixture`` columns.
struct FixtureSummary {
  double* xpts = nullptr;      // (P,)
  double* xpts_sd = nullptr;   // (P,) population std (ddof=0)
  double* p_played = nullptr;  // (P,)
  double* p_60 = nullptr;      // (P,)
  double* p_goal = nullptr;    // (P,)
  double* p_assist = nullptr;  // (P,)
  double* p_cs = nullptr;      // (P,)
  double* p_haul = nullptr;    // (P,) points >= 10
};

/// Aggregate (P, n) sim arrays into ``FixtureSummary`` (one pass, K5).
///
/// ``minutes`` / ``points`` / ``goals`` / ``assists`` / ``clean_sheet`` are
/// row-major ``(P, n)``. ``clean_sheet`` is 0/1 uint8.
void aggregate_fixture(const double* minutes, const double* points,
                       const std::int64_t* goals, const std::int64_t* assists,
                       const std::uint8_t* clean_sheet, std::size_t players,
                       std::size_t n, FixtureSummary* out) noexcept;

/// Simulation knobs shared with ``simulate_match`` (no hard-starters; K1 deferred).
struct SimParams {
  double own_goal_weight = 0.05;
  double p_assist = 0.75;
  double cs_red_void_prob = 0.0;
  bool conceded_scaled = true;  // false → "played"
  bool sample_other = false;
};

/// Injected uniforms for a full two-sided match (testing / L2 / L5).
struct MatchUniforms {
  const double* home_scorer = nullptr;       // (G_h, n)
  const double* home_assist_flag = nullptr;  // (G_h, n)
  const double* home_assister = nullptr;     // (G_h, n)
  std::size_t home_max_goals = 0;
  const double* away_scorer = nullptr;
  const double* away_assist_flag = nullptr;
  const double* away_assister = nullptr;
  std::size_t away_max_goals = 0;
  const double* cs_void = nullptr;  // (P, n) or null if unused
  const double* other = nullptr;    // (P, n) or null if unused
};

/// Optional (P, n) sample buffers; pass null to skip materialising a plane.
struct MatchSamples {
  std::int64_t* goals = nullptr;
  std::int64_t* assists = nullptr;
  double* points = nullptr;
  std::uint8_t* clean_sheets = nullptr;
  std::int64_t* unattributed_home = nullptr;  // (n,)
  std::int64_t* unattributed_away = nullptr;  // (n,)
};

/// Run allocate (both sides) + points (+ optional K4) with injected uniforms.
///
/// Minutes are taken as final (hard starters applied upstream in Python).
/// Writes into ``samples`` buffers when non-null; always fills ``summary`` when
/// non-null (aggregates from internal buffers if samples were not requested).
void simulate_match_with_uniforms(
    const std::int64_t* home_goals, const std::int64_t* away_goals,
    const std::uint8_t* side, const std::uint8_t* position, const double* minutes,
    const double* goal_weight, const double* assist_weight, const double* other_points,
    const RulesPOD& rules, const SimParams& params, const MatchUniforms& uniforms,
    const MatchOtherTables* match_other, std::size_t players, std::size_t n,
    MatchSamples* samples, FixtureSummary* summary) noexcept;

}  // namespace fplt_mc
