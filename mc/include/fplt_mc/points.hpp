#pragma once

#include <cstddef>
#include <cstdint>

#include "fplt_mc/rules.hpp"

namespace fplt_mc {

/// Match-conditioned other-points tables (K4).
///
/// Mirrors fitted ``OtherMatchConditioned`` deltas: per position, additive
/// offsets for team-goals band (0 / 1 / 2+), clean-sheet bit, and player-scored
/// bit. All zeros when unused. Delta is forced to 0 wherever minutes == 0.
struct MatchOtherTables {
  double delta_goals[kNPositions][3]{};
  double delta_cs[kNPositions][2]{};
  double delta_scored[kNPositions][2]{};
};

/// Inverse-CDF Poisson via sequential search (small-λ other-points path).
void poisson_sequential(const double* rate, const double* uniforms, std::size_t n,
                        std::int64_t* out) noexcept;

/// Additive match-conditioned other offset for one (player, sim).
[[nodiscard]] double match_delta(const MatchOtherTables& tables, std::uint8_t position,
                                 double minutes, std::int64_t team_goals, bool clean_sheet,
                                 std::int64_t player_goals) noexcept;

/// Score appearance, involvement, CS, conceded, and other points.
///
/// Matches Python ``simulation._points``. When ``match_other`` is non-null,
/// adds K4 table deltas to the other-points mean before sampling/expectation.
void points(const RulesPOD& rules, const std::uint8_t* position,
            const std::uint8_t* side, const double* minutes, const std::int64_t* goals,
            const std::int64_t* assists, const std::int64_t* home,
            const std::int64_t* away, const double* other_points,
            double cs_red_void_prob, bool conceded_scaled, bool sample_other,
            const double* cs_void_u, const double* other_u,
            const MatchOtherTables* match_other, std::size_t players, std::size_t n,
            double* points_out, std::uint8_t* clean_sheet_out) noexcept;

}  // namespace fplt_mc
