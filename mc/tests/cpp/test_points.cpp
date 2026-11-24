#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "fplt_mc/mathx.hpp"
#include "fplt_mc/points.hpp"
#include "fplt_mc/rules.hpp"

using fplt_mc::points;
using fplt_mc::poisson_sequential;
using fplt_mc::RulesPOD;

TEST(PoissonSequential, ZeroRateIsZero) {
  const double rate[] = {0.0, 0.0, 0.0};
  const double u[] = {0.1, 0.5, 0.9};
  std::int64_t out[3] = {9, 9, 9};
  poisson_sequential(rate, u, 3, out);
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[1], 0);
  EXPECT_EQ(out[2], 0);
}

TEST(PoissonSequential, SmallLambdaKnownDraw) {
  // λ=0 → always 0; for λ>0, u=0 → always k=0 (threshold never exceeds p0=e^{-λ}).
  const double rate[] = {1.5};
  const double u0[] = {0.0};
  std::int64_t out[1] = {};
  poisson_sequential(rate, u0, 1, out);
  EXPECT_EQ(out[0], 0);
}

TEST(PoissonSequential, MatchesExpCdfWalk) {
  const double lam = 0.8;
  const double u = 0.5;
  double prob = fplt_mc::mathx::exp(-lam);
  double cumulative = prob;
  std::int64_t expected = 0;
  while (u > cumulative) {
    ++expected;
    prob *= lam / static_cast<double>(expected);
    cumulative += prob;
  }
  const double rate[] = {lam};
  const double draw[] = {u};
  std::int64_t out[1] = {};
  poisson_sequential(rate, draw, 1, out);
  EXPECT_EQ(out[0], expected);
}

TEST(Points, AppearanceAndCleanSheet) {
  RulesPOD rules;
  rules.appearance_long_minutes = 60;
  rules.appearance_short = 1.0;
  rules.appearance_long = 2.0;
  rules.goals[0] = 6;
  rules.goals[1] = 6;
  rules.goals[2] = 5;
  rules.goals[3] = 4;
  rules.assists = 3.0;
  rules.clean_sheets[0] = 4;
  rules.clean_sheets[1] = 4;
  rules.clean_sheets[2] = 1;
  rules.clean_sheets[3] = 0;
  rules.goals_conceded_every = 2;
  rules.goals_conceded_points[0] = -1;
  rules.goals_conceded_points[1] = -1;
  rules.goals_conceded_points[2] = 0;
  rules.goals_conceded_points[3] = 0;

  // One DEF home, 3 sims: 90/45/90 mins; opp scores 0, 0, 1.
  const std::uint8_t position[] = {1};  // DEF
  const std::uint8_t side[] = {0};      // home
  const double minutes[] = {90.0, 45.0, 90.0};
  const std::int64_t goals[] = {0, 0, 0};
  const std::int64_t assists[] = {0, 0, 0};
  const std::int64_t home[] = {0, 0, 0};
  const std::int64_t away[] = {0, 0, 1};
  const double other[] = {0.0, 0.0, 0.0};
  double pts[3] = {};
  std::uint8_t cs[3] = {};
  points(rules, position, side, minutes, goals, assists, home, away, other,
         /*cs_red_void_prob=*/0.0, /*conceded_scaled=*/true,
         /*sample_other=*/false, nullptr, nullptr, nullptr, 1, 3, pts, cs);
  // CS + long appearance = 4+2=6; short app only = 1; long app no CS + conceded
  // for 1 goal: 2 + (-1)*(1//2)*1 = 2.
  EXPECT_DOUBLE_EQ(pts[0], 6.0);
  EXPECT_DOUBLE_EQ(pts[1], 1.0);
  EXPECT_DOUBLE_EQ(pts[2], 2.0);
  EXPECT_EQ(cs[0], 1);
  EXPECT_EQ(cs[1], 0);
  EXPECT_EQ(cs[2], 0);
}

TEST(Points, CsVoidWithInjectedUniform) {
  RulesPOD rules;
  const std::uint8_t position[] = {1};
  const std::uint8_t side[] = {0};
  const double minutes[] = {90.0};
  const std::int64_t goals[] = {0};
  const std::int64_t assists[] = {0};
  const std::int64_t home[] = {0};
  const std::int64_t away[] = {0};
  const double other[] = {0.0, 0.0, 0.0};
  const double void_u[] = {0.1};  // < 0.25 → void
  double pts[1] = {};
  std::uint8_t cs[1] = {};
  points(rules, position, side, minutes, goals, assists, home, away, other, 0.25, true,
         false, void_u, nullptr, nullptr, 1, 1, pts, cs);
  EXPECT_EQ(cs[0], 0);
  EXPECT_DOUBLE_EQ(pts[0], 2.0);  // appearance only
}

TEST(Points, ConcededScaledVsPlayed) {
  RulesPOD rules;
  rules.goals_conceded_points[1] = -1.0;
  rules.goals_conceded_every = 2;
  const std::uint8_t position[] = {1};
  const std::uint8_t side[] = {1};  // away
  const double minutes[] = {45.0};
  const std::int64_t z[] = {0};
  const std::int64_t home[] = {2};  // opp = 2 → 2//2 = 1
  const std::int64_t away_g[] = {0};
  const double other[] = {0.0, 0.0, 0.0};
  double pts_scaled[1] = {};
  double pts_played[1] = {};
  std::uint8_t cs[1] = {};
  points(rules, position, side, minutes, z, z, home, away_g, other, 0.0, true, false,
         nullptr, nullptr, nullptr, 1, 1, pts_scaled, cs);
  points(rules, position, side, minutes, z, z, home, away_g, other, 0.0, false, false,
         nullptr, nullptr, nullptr, 1, 1, pts_played, cs);
  // scaled: 1 (short) + (-1)*1*(45/90) = 0.5
  // played: 1 + (-1)*1*1 = 0
  EXPECT_DOUBLE_EQ(pts_scaled[0], 0.5);
  EXPECT_DOUBLE_EQ(pts_played[0], 0.0);
}
