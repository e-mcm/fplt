#pragma once

#include <cstddef>
#include <cstdint>

namespace fplt_mc {

/// Pick one player index per simulation via inverse CDF over player-row order.
///
/// Matches Python ``simulation._pick``: the first index where the running sum
/// exceeds ``u · total`` (``total = player mass + extra``); ``-1`` if none, or
/// if ``total ≤ 0``. Summation order is player-row order (``np.cumsum``).
///
/// Args:
///   weights: Row-major ``(players, n)`` rates (may be zero).
///   draw: Length-``n`` uniforms in ``[0, 1)``.
///   extra: Non-player mass (own-goal bucket when > 0; assist path uses 0).
///   players: Number of player rows.
///   n: Number of simulations.
///   out: Length-``n`` int64 indices written in place.
void pick(const double* weights, const double* draw, double extra, std::size_t players,
          std::size_t n, std::int64_t* out) noexcept;

/// Allocate goals and assists for one side using injected uniforms.
///
/// Matches Python ``simulation._allocate_side`` with ``SideAllocateUniforms``:
/// rates are ``weight · minutes/90`` when minutes > 0 else 0; for each goal
/// row until no sim still needs a goal, pick scorer (extra = own-goal weight),
/// then optionally an assister with the scorer's assist rate zeroed.
///
/// Uniform grids ``scorer_u``, ``assist_flag_u``, ``assister_u`` are row-major
/// ``(max_goals, n)`` with ``max_goals >= max(team_goals)`` (caller-validated).
/// Early-break semantics match Python (unused trailing goal rows ignored).
///
/// Args:
///   team_goals: Length-``n`` goals scored by this side per sim.
///   minutes: Row-major ``(players, n)``.
///   goal_weight / assist_weight: Length-``players``.
///   own_goal_weight: Extra mass for the unattributed scorer bucket.
///   p_assist: Probability a claimed goal also gets an assist.
///   scorer_u / assist_flag_u / assister_u: Injected uniforms.
///   players / n / max_goals: Shapes.
///   scored / assisted: Row-major ``(players, n)`` int64 outputs (zeroed first).
///   missed: Length-``n`` unattributed goals (zeroed first).
void allocate_side(const std::int64_t* team_goals, const double* minutes,
                   const double* goal_weight, const double* assist_weight,
                   double own_goal_weight, double p_assist, const double* scorer_u,
                   const double* assist_flag_u, const double* assister_u,
                   std::size_t players, std::size_t n, std::size_t max_goals,
                   std::int64_t* scored, std::int64_t* assisted,
                   std::int64_t* missed) noexcept;

}  // namespace fplt_mc
