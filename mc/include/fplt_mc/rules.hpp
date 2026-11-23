#pragma once

#include <cstddef>
#include <cstdint>

namespace fplt_mc {

/// Position indices matching the native fixture layout (GK=0 … FWD=3).
inline constexpr std::size_t kNPositions = 4;

/// Flattened FPL scoring rules for the points kernel (no YAML in C++).
///
/// Built from Python ``Rules`` at the binding boundary. Goal / CS / conceded
/// tables are indexed by position ``0=GK, 1=DEF, 2=MID, 3=FWD``.
struct RulesPOD {
  std::int32_t appearance_long_minutes = 60;
  double appearance_short = 1.0;
  double appearance_long = 2.0;
  double goals[kNPositions] = {6.0, 6.0, 5.0, 4.0};
  double assists = 3.0;
  double clean_sheets[kNPositions] = {4.0, 4.0, 1.0, 0.0};
  std::int32_t goals_conceded_every = 2;
  /// Per-position conceded points (0 when the position is not in the rule).
  double goals_conceded_points[kNPositions] = {-1.0, -1.0, 0.0, 0.0};
};

/// Side / position codes for the points kernel.
inline constexpr std::uint8_t kSideHome = 0;
inline constexpr std::uint8_t kSideAway = 1;
inline constexpr std::uint8_t kPosGK = 0;
inline constexpr std::uint8_t kPosDEF = 1;
inline constexpr std::uint8_t kPosMID = 2;
inline constexpr std::uint8_t kPosFWD = 3;

}  // namespace fplt_mc
