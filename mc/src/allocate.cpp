#include "fplt_mc/allocate.hpp"

#include <cstring>
#include <vector>

namespace fplt_mc {
namespace {

[[nodiscard]] inline std::size_t idx(std::size_t row, std::size_t col,
                                     std::size_t ncols) noexcept {
  return row * ncols + col;
}

}  // namespace

void pick(const double* weights, const double* draw, double extra, std::size_t players,
          std::size_t n, std::int64_t* out) noexcept {
  for (std::size_t j = 0; j < n; ++j) {
    double player_mass = 0.0;
    for (std::size_t i = 0; i < players; ++i) {
      player_mass += weights[idx(i, j, n)];
    }
    const double total = player_mass + extra;
    if (!(total > 0.0)) {
      out[j] = -1;
      continue;
    }
    const double threshold = draw[j] * total;
    double running = 0.0;
    std::int64_t chosen = -1;
    for (std::size_t i = 0; i < players; ++i) {
      running += weights[idx(i, j, n)];
      if (running > threshold) {
        chosen = static_cast<std::int64_t>(i);
        break;
      }
    }
    out[j] = chosen;
  }
}

void allocate_side(const std::int64_t* team_goals, const double* minutes,
                   const double* goal_weight, const double* assist_weight,
                   double own_goal_weight, double p_assist, const double* scorer_u,
                   const double* assist_flag_u, const double* assister_u,
                   std::size_t players, std::size_t n, std::size_t max_goals,
                   std::int64_t* scored, std::int64_t* assisted,
                   std::int64_t* missed) noexcept {
  const std::size_t pn = players * n;
  std::memset(scored, 0, pn * sizeof(std::int64_t));
  std::memset(assisted, 0, pn * sizeof(std::int64_t));
  std::memset(missed, 0, n * sizeof(std::int64_t));

  if (n == 0 || max_goals == 0) {
    return;
  }
  if (players == 0) {
    // Python: return empty scored/assisted and team_goals.copy() for missed.
    std::memcpy(missed, team_goals, n * sizeof(std::int64_t));
    return;
  }

  std::vector<double> goal_rates(pn);
  std::vector<double> assist_rates(pn);
  for (std::size_t i = 0; i < players; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double m = minutes[idx(i, j, n)];
      const std::size_t k = idx(i, j, n);
      if (m > 0.0) {
        goal_rates[k] = goal_weight[i] * m / 90.0;
        assist_rates[k] = assist_weight[i] * m / 90.0;
      } else {
        goal_rates[k] = 0.0;
        assist_rates[k] = 0.0;
      }
    }
  }

  std::vector<std::int64_t> scored_sum(n, 0);
  std::vector<std::int64_t> scorer(n);
  std::vector<std::int64_t> assister_idx(n);
  std::vector<double> blocked(pn);
  std::vector<char> active(n);

  for (std::size_t g = 0; g < max_goals; ++g) {
    bool any_active = false;
    for (std::size_t j = 0; j < n; ++j) {
      active[j] = static_cast<char>(team_goals[j] > scored_sum[j] + missed[j]);
      any_active = any_active || active[j];
    }
    if (!any_active) {
      break;
    }

    const double* draw = scorer_u + g * n;
    const double* assist_draw = assist_flag_u + g * n;
    const double* second = assister_u + g * n;

    pick(goal_rates.data(), draw, own_goal_weight, players, n, scorer.data());

    for (std::size_t j = 0; j < n; ++j) {
      if (!active[j]) {
        continue;
      }
      if (scorer[j] >= 0) {
        const std::size_t p = static_cast<std::size_t>(scorer[j]);
        scored[idx(p, j, n)] += 1;
        scored_sum[j] += 1;
      } else {
        missed[j] += 1;
      }
    }

    // Match NumPy: zero assist rate at max(scorer, 0) when scorer >= 0,
    // for every sim (including inactive).
    std::memcpy(blocked.data(), assist_rates.data(), pn * sizeof(double));
    for (std::size_t j = 0; j < n; ++j) {
      if (scorer[j] >= 0) {
        const std::size_t p = static_cast<std::size_t>(scorer[j]);
        blocked[idx(p, j, n)] = 0.0;
      }
    }

    pick(blocked.data(), second, /*extra=*/0.0, players, n, assister_idx.data());

    for (std::size_t j = 0; j < n; ++j) {
      const bool claimed = active[j] && scorer[j] >= 0;
      const bool want_assist = claimed && (assist_draw[j] < p_assist);
      if (want_assist && assister_idx[j] >= 0) {
        const std::size_t p = static_cast<std::size_t>(assister_idx[j]);
        assisted[idx(p, j, n)] += 1;
      }
    }
  }
}

}  // namespace fplt_mc
