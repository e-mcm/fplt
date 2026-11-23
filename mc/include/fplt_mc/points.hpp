#pragma once

#include <cstddef>
#include <cstdint>

#include "fplt_mc/rules.hpp"

namespace fplt_mc {

/// Inverse-CDF Poisson via sequential search (small-λ other-points path).
///
/// Matches Python ``simulation._poisson_sequential``: for each cell with
/// ``λ > 0``, start from ``p = exp(−λ)`` and walk ``k = 0, 1, …`` until the
/// running CDF exceeds ``u``. ``λ ≤ 0`` yields 0. Uses ``mathx::exp``.
///
/// Args:
///   rate / uniforms: Length-``n`` non-negative λ and U(0,1) draws.
///   out: Length-``n`` int64 counts written in place.
void poisson_sequential(const double* rate, const double* uniforms, std::size_t n,
                        std::int64_t* out) noexcept;

/// Score appearance, involvement, CS, conceded, and other points.
///
/// Matches Python ``simulation._points`` with injected CS-void / other
/// uniforms and ``match_other=None`` (K4 match-delta lands later).
///
/// Args:
///   rules: Flattened scoring table.
///   position: Length-``P`` codes ``0=GK…3=FWD``.
///   side: Length-``P`` codes ``0=home, 1=away``.
///   minutes / goals / assists: Row-major ``(P, n)``.
///   home / away: Length-``n`` team goals.
///   other_points: Row-major ``(P, 3)`` bucket means (0 / 1–59 / 60+).
///   cs_red_void_prob: Void probability when a CS would otherwise hold.
///   conceded_scaled: ``true`` → minutes/90 exposure; ``false`` → played.
///   sample_other: ``true`` → sequential-search Poisson from ``other_u``.
///   cs_void_u: Row-major ``(P, n)`` (required when ``cs_red_void_prob > 0``).
///   other_u: Row-major ``(P, n)`` (required when ``sample_other``).
///   players / n: Shapes.
///   points_out: Row-major ``(P, n)`` float64.
///   clean_sheet_out: Row-major ``(P, n)`` uint8 (0/1 after void).
void points(const RulesPOD& rules, const std::uint8_t* position,
            const std::uint8_t* side, const double* minutes, const std::int64_t* goals,
            const std::int64_t* assists, const std::int64_t* home,
            const std::int64_t* away, const double* other_points,
            double cs_red_void_prob, bool conceded_scaled, bool sample_other,
            const double* cs_void_u, const double* other_u, std::size_t players,
            std::size_t n, double* points_out, std::uint8_t* clean_sheet_out) noexcept;

}  // namespace fplt_mc
