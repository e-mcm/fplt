#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "fplt_mc/allocate.hpp"

using fplt_mc::allocate_side;
using fplt_mc::pick;

TEST(Pick, ZeroMassReturnsMinusOne) {
  // players=2, n=1, all-zero weights, extra=0
  const double weights[] = {0.0, 0.0};
  const double draw[] = {0.3};
  std::int64_t out[1] = {99};
  pick(weights, draw, /*extra=*/0.0, /*players=*/2, /*n=*/1, out);
  EXPECT_EQ(out[0], -1);
}

TEST(Pick, OnlyExtraBucket) {
  // No player mass; extra > 0 → always -1 (own-goal / unattributed).
  // Row-major (2 players, 3 sims) — six zeros.
  const double weights[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const double draw[] = {0.0, 0.5, 0.999};
  std::int64_t out[3] = {};
  pick(weights, draw, /*extra=*/1.0, /*players=*/2, /*n=*/3, out);
  EXPECT_EQ(out[0], -1);
  EXPECT_EQ(out[1], -1);
  EXPECT_EQ(out[2], -1);
}

TEST(Pick, SinglePlayerAlwaysChosenWhenExtraZero) {
  // weights (1, 3): player 0 has mass 1.0 every sim; extra=0.
  const double weights[] = {1.0, 1.0, 1.0};
  const double draw[] = {0.0, 0.5, 0.999999};
  std::int64_t out[3] = {};
  pick(weights, draw, /*extra=*/0.0, /*players=*/1, /*n=*/3, out);
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[1], 0);
  EXPECT_EQ(out[2], 0);
}

TEST(Pick, PlayerRowOrderMatchesCumsum) {
  // Two players, one sim: masses 0.25, 0.75; extra=0. Threshold = u * 1.0.
  // u=0.2 → first cumsum 0.25 > 0.2 → index 0
  // u=0.3 → 0.25 not >, 1.0 > 0.3 → index 1
  const double weights[] = {0.25, 0.75};  // row-major (2, 1)
  const double draw0[] = {0.2};
  const double draw1[] = {0.3};
  std::int64_t out0[1] = {};
  std::int64_t out1[1] = {};
  pick(weights, draw0, 0.0, 2, 1, out0);
  pick(weights, draw1, 0.0, 2, 1, out1);
  EXPECT_EQ(out0[0], 0);
  EXPECT_EQ(out1[0], 1);
}

TEST(Pick, ExtraBucketWhenDrawInTail) {
  // player mass 1.0, extra 1.0, total 2.0. u=0.6 → threshold 1.2 > 1.0 → -1
  const double weights[] = {1.0};
  const double draw[] = {0.6};
  std::int64_t out[1] = {};
  pick(weights, draw, /*extra=*/1.0, 1, 1, out);
  EXPECT_EQ(out[0], -1);
}

TEST(AllocateSide, EmptyGoalsNoOps) {
  const std::int64_t team_goals[] = {0, 0};
  const double minutes[] = {90.0, 90.0, 90.0, 90.0};  // 2 players × 2 sims
  const double gw[] = {1.0, 1.0};
  const double aw[] = {1.0, 1.0};
  std::int64_t scored[4];
  std::int64_t assisted[4];
  std::int64_t missed[2];
  // max_goals=0 → early return, zeros
  allocate_side(team_goals, minutes, gw, aw, 0.05, 0.8, nullptr, nullptr, nullptr, 2, 2,
                0, scored, assisted, missed);
  EXPECT_EQ(scored[0], 0);
  EXPECT_EQ(missed[0], 0);
  EXPECT_EQ(missed[1], 0);
}

TEST(AllocateSide, ZeroMinutePlayersNeverScore) {
  // One sim, 2 goals, only player 1 has minutes; own_goal_weight=0.
  const std::int64_t team_goals[] = {2};
  const double minutes[] = {0.0, 90.0};  // (2, 1)
  const double gw[] = {1.0, 1.0};
  const double aw[] = {1.0, 1.0};
  // Two goal rows of uniforms: always pick into player mass.
  const double scorer_u[] = {0.1, 0.1};
  const double flag_u[] = {0.0, 0.0};  // always want assist
  const double ast_u[] = {0.5, 0.5};
  std::int64_t scored[2] = {};
  std::int64_t assisted[2] = {};
  std::int64_t missed[1] = {};
  allocate_side(team_goals, minutes, gw, aw, 0.0, 1.0, scorer_u, flag_u, ast_u, 2, 1, 2,
                scored, assisted, missed);
  EXPECT_EQ(scored[0], 0);  // player 0
  EXPECT_EQ(scored[1], 2);  // player 1
  EXPECT_EQ(missed[0], 0);
  // Only one on-pitch player → assister pick has zero mass after zeroing scorer
  EXPECT_EQ(assisted[0], 0);
  EXPECT_EQ(assisted[1], 0);
}

TEST(AllocateSide, GoalsPlusMissedEqualScoreline) {
  const std::int64_t team_goals[] = {3};
  const double minutes[] = {90.0, 90.0};
  const double gw[] = {1.0, 0.0};
  const double aw[] = {0.0, 1.0};
  // Mix of claimed and own-goal: u near 1 with equal mass+extra → -1.
  // player mass=1 (only p0), extra=1, total=2. u=0.9 → thresh 1.8 → -1
  // u=0.1 → thresh 0.2 → player 0
  const double scorer_u[] = {0.1, 0.9, 0.1};
  const double flag_u[] = {0.0, 0.0, 0.0};
  const double ast_u[] = {0.0, 0.0, 0.0};
  std::int64_t scored[2] = {};
  std::int64_t assisted[2] = {};
  std::int64_t missed[1] = {};
  allocate_side(team_goals, minutes, gw, aw, 1.0, 1.0, scorer_u, flag_u, ast_u, 2, 1, 3,
                scored, assisted, missed);
  EXPECT_EQ(scored[0] + scored[1] + missed[0], 3);
  EXPECT_EQ(assisted[0], 0);  // scorer is p0; assist weight of p0 is 0
  EXPECT_EQ(assisted[1], 2);  // two claimed goals, assister is p1
}
