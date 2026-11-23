#include "fplt_mc/points.hpp"

#include <vector>

#include "fplt_mc/mathx.hpp"

namespace fplt_mc {
namespace {

[[nodiscard]] inline std::size_t idx(std::size_t row, std::size_t col,
                                     std::size_t ncols) noexcept {
  return row * ncols + col;
}

}  // namespace

void poisson_sequential(const double* rate, const double* uniforms, std::size_t n,
                        std::int64_t* out) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    const double intensity = rate[i];
    if (!(intensity > 0.0)) {
      out[i] = 0;
      continue;
    }
    const double threshold = uniforms[i];
    double prob = mathx::exp(-intensity);
    double cumulative = prob;
    std::int64_t count = 0;
    while (threshold > cumulative) {
      ++count;
      prob *= intensity / static_cast<double>(count);
      cumulative += prob;
    }
    out[i] = count;
  }
}

double match_delta(const MatchOtherTables& tables, std::uint8_t position,
                   double minutes, std::int64_t team_goals, bool clean_sheet,
                   std::int64_t player_goals) noexcept {
  if (!(minutes > 0.0) || position >= kNPositions) {
    return 0.0;
  }
  const std::size_t band = team_goals <= 0 ? 0 : (team_goals == 1 ? 1 : 2);
  const std::size_t cs_i = clean_sheet ? 1 : 0;
  const std::size_t scored_i = player_goals >= 1 ? 1 : 0;
  return tables.delta_goals[position][band] + tables.delta_cs[position][cs_i] +
         tables.delta_scored[position][scored_i];
}

void points(const RulesPOD& rules, const std::uint8_t* position,
            const std::uint8_t* side, const double* minutes, const std::int64_t* goals,
            const std::int64_t* assists, const std::int64_t* home,
            const std::int64_t* away, const double* other_points,
            double cs_red_void_prob, bool conceded_scaled, bool sample_other,
            const double* cs_void_u, const double* other_u,
            const MatchOtherTables* match_other, std::size_t players, std::size_t n,
            double* points_out, std::uint8_t* clean_sheet_out) noexcept {
  const std::size_t pn = players * n;
  if (pn == 0) {
    return;
  }
  const double long_minutes = static_cast<double>(rules.appearance_long_minutes);
  const std::int64_t every =
      rules.goals_conceded_every > 0 ? rules.goals_conceded_every : 1;

  std::vector<double> other_mean(pn);

  for (std::size_t i = 0; i < players; ++i) {
    const std::size_t pos = position[i] < kNPositions ? position[i] : 0;
    const bool is_home = side[i] == kSideHome;
    const double goal_value = rules.goals[pos];
    const double sheet_value = rules.clean_sheets[pos];
    const double conceded_value = rules.goals_conceded_points[pos];

    for (std::size_t j = 0; j < n; ++j) {
      const std::size_t k = idx(i, j, n);
      const double m = minutes[k];
      const std::int64_t opp = is_home ? away[j] : home[j];
      const std::int64_t team_g = is_home ? home[j] : away[j];

      double appearance = 0.0;
      if (m >= long_minutes) {
        appearance = rules.appearance_long;
      } else if (m > 0.0) {
        appearance = rules.appearance_short;
      }

      bool cs = (opp == 0) && (m >= long_minutes);
      if (cs && cs_red_void_prob > 0.0 && cs_void_u != nullptr) {
        if (cs_void_u[k] < cs_red_void_prob) {
          cs = false;
        }
      }
      clean_sheet_out[k] = static_cast<std::uint8_t>(cs ? 1 : 0);

      double exposure = 0.0;
      if (m > 0.0) {
        exposure = conceded_scaled ? (m / 90.0) : 1.0;
      }
      const double conceded =
          conceded_value * static_cast<double>(opp / every) * exposure;

      std::size_t bucket = 0;
      if (m > 0.0) {
        bucket = (m < 60.0) ? 1 : 2;
      }
      other_mean[k] = other_points[idx(i, bucket, 3)];
      if (match_other != nullptr) {
        other_mean[k] += match_delta(*match_other, static_cast<std::uint8_t>(pos), m,
                                     team_g, cs, goals[k]);
      }

      points_out[k] = appearance + goal_value * static_cast<double>(goals[k]) +
                      rules.assists * static_cast<double>(assists[k]) +
                      sheet_value * (cs ? 1.0 : 0.0) + conceded;
    }
  }

  if (sample_other && other_u != nullptr) {
    std::vector<double> rate(pn);
    for (std::size_t k = 0; k < pn; ++k) {
      rate[k] = other_mean[k] >= 0.0 ? other_mean[k] : -other_mean[k];
    }
    std::vector<std::int64_t> drawn(pn);
    poisson_sequential(rate.data(), other_u, pn, drawn.data());
    for (std::size_t k = 0; k < pn; ++k) {
      const double signed_other = other_mean[k] >= 0.0 ? static_cast<double>(drawn[k])
                                                       : -static_cast<double>(drawn[k]);
      points_out[k] += signed_other;
    }
  } else {
    for (std::size_t k = 0; k < pn; ++k) {
      points_out[k] += other_mean[k];
    }
  }
}

}  // namespace fplt_mc
