#include "fplt_mc/simulate.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "fplt_mc/allocate.hpp"
#include "fplt_mc/rng.hpp"

namespace fplt_mc {
namespace {

[[nodiscard]] inline std::size_t idx(std::size_t row, std::size_t col,
                                     std::size_t ncols) noexcept {
  return row * ncols + col;
}

void aggregate_one_player(const double* minutes_row, const double* points_row,
                          const std::int64_t* goals_row,
                          const std::int64_t* assists_row, const std::uint8_t* cs_row,
                          std::size_t n, double* xpts, double* xpts_sd,
                          double* p_played, double* p_60, double* p_goal,
                          double* p_assist, double* p_cs, double* p_haul) noexcept {
  if (n == 0) {
    if (xpts != nullptr) {
      *xpts = 0.0;
    }
    if (xpts_sd != nullptr) {
      *xpts_sd = 0.0;
    }
    if (p_played != nullptr) {
      *p_played = 0.0;
    }
    if (p_60 != nullptr) {
      *p_60 = 0.0;
    }
    if (p_goal != nullptr) {
      *p_goal = 0.0;
    }
    if (p_assist != nullptr) {
      *p_assist = 0.0;
    }
    if (p_cs != nullptr) {
      *p_cs = 0.0;
    }
    if (p_haul != nullptr) {
      *p_haul = 0.0;
    }
    return;
  }
  double sum = 0.0;
  double sum_sq = 0.0;
  std::size_t played = 0;
  std::size_t ge60 = 0;
  std::size_t ge_goal = 0;
  std::size_t ge_assist = 0;
  std::size_t cs_count = 0;
  std::size_t haul = 0;
  for (std::size_t j = 0; j < n; ++j) {
    const double p = points_row[j];
    sum += p;
    sum_sq += p * p;
    if (minutes_row[j] > 0.0) {
      ++played;
    }
    if (minutes_row[j] >= 60.0) {
      ++ge60;
    }
    if (goals_row[j] >= 1) {
      ++ge_goal;
    }
    if (assists_row[j] >= 1) {
      ++ge_assist;
    }
    if (cs_row[j] != 0) {
      ++cs_count;
    }
    if (p >= 10.0) {
      ++haul;
    }
  }
  const double nn = static_cast<double>(n);
  const double mean = sum / nn;
  // Population std (numpy default ddof=0); 0 when n==1.
  const double var = sum_sq / nn - mean * mean;
  const double sd = var > 0.0 ? std::sqrt(var) : 0.0;
  if (xpts != nullptr) {
    *xpts = mean;
  }
  if (xpts_sd != nullptr) {
    *xpts_sd = sd;
  }
  if (p_played != nullptr) {
    *p_played = static_cast<double>(played) / nn;
  }
  if (p_60 != nullptr) {
    *p_60 = static_cast<double>(ge60) / nn;
  }
  if (p_goal != nullptr) {
    *p_goal = static_cast<double>(ge_goal) / nn;
  }
  if (p_assist != nullptr) {
    *p_assist = static_cast<double>(ge_assist) / nn;
  }
  if (p_cs != nullptr) {
    *p_cs = static_cast<double>(cs_count) / nn;
  }
  if (p_haul != nullptr) {
    *p_haul = static_cast<double>(haul) / nn;
  }
}

}  // namespace

void aggregate_fixture(const double* minutes, const double* points,
                       const std::int64_t* goals, const std::int64_t* assists,
                       const std::uint8_t* clean_sheet, std::size_t players,
                       std::size_t n, FixtureSummary* out) noexcept {
  if (out == nullptr) {
    return;
  }
  for (std::size_t i = 0; i < players; ++i) {
    const std::size_t row = i * n;
    aggregate_one_player(
        minutes + row, points + row, goals + row, assists + row, clean_sheet + row, n,
        out->xpts ? out->xpts + i : nullptr, out->xpts_sd ? out->xpts_sd + i : nullptr,
        out->p_played ? out->p_played + i : nullptr,
        out->p_60 ? out->p_60 + i : nullptr, out->p_goal ? out->p_goal + i : nullptr,
        out->p_assist ? out->p_assist + i : nullptr,
        out->p_cs ? out->p_cs + i : nullptr, out->p_haul ? out->p_haul + i : nullptr);
  }
}

void simulate_match_with_uniforms(
    const std::int64_t* home_goals, const std::int64_t* away_goals,
    const std::uint8_t* side, const std::uint8_t* position, const double* minutes,
    const double* goal_weight, const double* assist_weight, const double* other_points,
    const RulesPOD& rules, const SimParams& params, const MatchUniforms& uniforms,
    const MatchOtherTables* match_other, std::size_t players, std::size_t n,
    MatchSamples* samples, FixtureSummary* summary) noexcept {
  const std::size_t pn = players * n;
  std::vector<std::int64_t> goals_buf(pn, 0);
  std::vector<std::int64_t> assists_buf(pn, 0);
  std::vector<double> points_buf(pn, 0.0);
  std::vector<std::uint8_t> cs_buf(pn, 0);
  std::vector<std::int64_t> unattr_home(n, 0);
  std::vector<std::int64_t> unattr_away(n, 0);

  // Count home / away players and build contiguous side packs for allocate.
  std::vector<std::size_t> home_idx;
  std::vector<std::size_t> away_idx;
  home_idx.reserve(players);
  away_idx.reserve(players);
  for (std::size_t i = 0; i < players; ++i) {
    if (side[i] == kSideHome) {
      home_idx.push_back(i);
    } else {
      away_idx.push_back(i);
    }
  }

  auto run_side = [&](const std::vector<std::size_t>& idxs,
                      const std::int64_t* team_goals, const double* scorer_u,
                      const double* flag_u, const double* ast_u, std::size_t max_goals,
                      std::int64_t* missed_out) {
    const std::size_t sp = idxs.size();
    if (sp == 0) {
      if (missed_out != nullptr) {
        std::memcpy(missed_out, team_goals, n * sizeof(std::int64_t));
      }
      return;
    }
    std::vector<double> side_minutes(sp * n);
    std::vector<double> side_gw(sp);
    std::vector<double> side_aw(sp);
    for (std::size_t s = 0; s < sp; ++s) {
      const std::size_t i = idxs[s];
      side_gw[s] = goal_weight[i];
      side_aw[s] = assist_weight[i];
      for (std::size_t j = 0; j < n; ++j) {
        side_minutes[idx(s, j, n)] = minutes[idx(i, j, n)];
      }
    }
    std::vector<std::int64_t> side_scored(sp * n);
    std::vector<std::int64_t> side_assisted(sp * n);
    std::vector<std::int64_t> side_missed(n);
    allocate_side(team_goals, side_minutes.data(), side_gw.data(), side_aw.data(),
                  params.own_goal_weight, params.p_assist, scorer_u, flag_u, ast_u, sp,
                  n, max_goals, side_scored.data(), side_assisted.data(),
                  side_missed.data());
    for (std::size_t s = 0; s < sp; ++s) {
      const std::size_t i = idxs[s];
      for (std::size_t j = 0; j < n; ++j) {
        goals_buf[idx(i, j, n)] = side_scored[idx(s, j, n)];
        assists_buf[idx(i, j, n)] = side_assisted[idx(s, j, n)];
      }
    }
    if (missed_out != nullptr) {
      std::memcpy(missed_out, side_missed.data(), n * sizeof(std::int64_t));
    }
  };

  run_side(home_idx, home_goals, uniforms.home_scorer, uniforms.home_assist_flag,
           uniforms.home_assister, uniforms.home_max_goals, unattr_home.data());
  run_side(away_idx, away_goals, uniforms.away_scorer, uniforms.away_assist_flag,
           uniforms.away_assister, uniforms.away_max_goals, unattr_away.data());

  points(rules, position, side, minutes, goals_buf.data(), assists_buf.data(),
         home_goals, away_goals, other_points, params.cs_red_void_prob,
         params.conceded_scaled, params.sample_other, uniforms.cs_void, uniforms.other,
         match_other, players, n, points_buf.data(), cs_buf.data());

  if (samples != nullptr) {
    if (samples->goals != nullptr) {
      std::memcpy(samples->goals, goals_buf.data(), pn * sizeof(std::int64_t));
    }
    if (samples->assists != nullptr) {
      std::memcpy(samples->assists, assists_buf.data(), pn * sizeof(std::int64_t));
    }
    if (samples->points != nullptr) {
      std::memcpy(samples->points, points_buf.data(), pn * sizeof(double));
    }
    if (samples->clean_sheets != nullptr) {
      std::memcpy(samples->clean_sheets, cs_buf.data(), pn * sizeof(std::uint8_t));
    }
    if (samples->unattributed_home != nullptr) {
      std::memcpy(samples->unattributed_home, unattr_home.data(),
                  n * sizeof(std::int64_t));
    }
    if (samples->unattributed_away != nullptr) {
      std::memcpy(samples->unattributed_away, unattr_away.data(),
                  n * sizeof(std::int64_t));
    }
  }

  if (summary != nullptr) {
    aggregate_fixture(minutes, points_buf.data(), goals_buf.data(), assists_buf.data(),
                      cs_buf.data(), players, n, summary);
  }
}

namespace {

void fill_side_uniforms(std::uint64_t kernel_seed, std::uint32_t stream_scorer,
                        std::uint32_t stream_flag, std::uint32_t stream_assister,
                        std::size_t max_goals, std::size_t n, double* scorer,
                        double* flag, double* assister) noexcept {
  for (std::size_t g = 0; g < max_goals; ++g) {
    for (std::size_t j = 0; j < n; ++j) {
      const std::size_t k = g * n + j;
      const auto sim = static_cast<std::uint32_t>(j);
      const auto slot = static_cast<std::uint32_t>(g);
      scorer[k] = philox_u01(kernel_seed, stream_scorer, sim, slot);
      flag[k] = philox_u01(kernel_seed, stream_flag, sim, slot);
      assister[k] = philox_u01(kernel_seed, stream_assister, sim, slot);
    }
  }
}

[[nodiscard]] std::size_t max_goals_of(const std::int64_t* goals,
                                       std::size_t n) noexcept {
  std::int64_t m = 0;
  for (std::size_t j = 0; j < n; ++j) {
    m = std::max(m, goals[j]);
  }
  return m > 0 ? static_cast<std::size_t>(m) : 0;
}

}  // namespace

void simulate_fixture(const std::int64_t* home_goals, const std::int64_t* away_goals,
                      const std::uint8_t* side, const std::uint8_t* position,
                      const double* minutes, const double* goal_weight,
                      const double* assist_weight, const double* other_points,
                      const RulesPOD& rules, const SimParams& params,
                      const MatchOtherTables* match_other, std::uint64_t kernel_seed,
                      std::size_t players, std::size_t n, MatchSamples* samples,
                      FixtureSummary* summary) noexcept {
  const std::size_t home_max = max_goals_of(home_goals, n);
  const std::size_t away_max = max_goals_of(away_goals, n);
  const std::size_t pn = players * n;

  std::vector<double> home_scorer(home_max * n);
  std::vector<double> home_flag(home_max * n);
  std::vector<double> home_ast(home_max * n);
  std::vector<double> away_scorer(away_max * n);
  std::vector<double> away_flag(away_max * n);
  std::vector<double> away_ast(away_max * n);
  fill_side_uniforms(kernel_seed, kStreamHomeScorer, kStreamHomeAssistFlag,
                     kStreamHomeAssister, home_max, n, home_scorer.data(),
                     home_flag.data(), home_ast.data());
  fill_side_uniforms(kernel_seed, kStreamAwayScorer, kStreamAwayAssistFlag,
                     kStreamAwayAssister, away_max, n, away_scorer.data(),
                     away_flag.data(), away_ast.data());

  std::vector<double> cs_void;
  std::vector<double> other_u;
  if (params.cs_red_void_prob > 0.0) {
    cs_void.resize(pn);
    for (std::size_t i = 0; i < players; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        cs_void[i * n + j] =
            philox_u01(kernel_seed, kStreamCsVoid, static_cast<std::uint32_t>(j),
                       static_cast<std::uint32_t>(i));
      }
    }
  }
  if (params.sample_other) {
    other_u.resize(pn);
    for (std::size_t i = 0; i < players; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        other_u[i * n + j] =
            philox_u01(kernel_seed, kStreamOther, static_cast<std::uint32_t>(j),
                       static_cast<std::uint32_t>(i));
      }
    }
  }

  MatchUniforms uniforms;
  uniforms.home_scorer = home_scorer.data();
  uniforms.home_assist_flag = home_flag.data();
  uniforms.home_assister = home_ast.data();
  uniforms.home_max_goals = home_max;
  uniforms.away_scorer = away_scorer.data();
  uniforms.away_assist_flag = away_flag.data();
  uniforms.away_assister = away_ast.data();
  uniforms.away_max_goals = away_max;
  uniforms.cs_void = cs_void.empty() ? nullptr : cs_void.data();
  uniforms.other = other_u.empty() ? nullptr : other_u.data();

  simulate_match_with_uniforms(home_goals, away_goals, side, position, minutes,
                               goal_weight, assist_weight, other_points, rules, params,
                               uniforms, match_other, players, n, samples, summary);
}

}  // namespace fplt_mc
