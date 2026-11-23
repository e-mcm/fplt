#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "fplt_mc/points.hpp"
#include "fplt_mc/rules.hpp"
#include "fplt_mc/simulate.hpp"

using fplt_mc::aggregate_fixture;
using fplt_mc::FixtureSummary;
using fplt_mc::match_delta;
using fplt_mc::MatchOtherTables;
using fplt_mc::MatchSamples;
using fplt_mc::MatchUniforms;
using fplt_mc::RulesPOD;
using fplt_mc::SimParams;
using fplt_mc::simulate_match_with_uniforms;

TEST(MatchDelta, ZeroMinutesIsZero) {
  MatchOtherTables tables{};
  tables.delta_goals[1][2] = 1.5;
  tables.delta_cs[1][1] = 0.5;
  tables.delta_scored[1][1] = 0.25;
  EXPECT_DOUBLE_EQ(match_delta(tables, 1, 0.0, 3, true, 2), 0.0);
  EXPECT_DOUBLE_EQ(match_delta(tables, 1, 90.0, 3, true, 2), 2.25);
  EXPECT_DOUBLE_EQ(match_delta(tables, 1, 90.0, 1, false, 0), 0.0);
}

TEST(Aggregate, MeanStdAndProbabilities) {
  // One player, 4 sims.
  const double minutes[] = {0.0, 45.0, 60.0, 90.0};
  const double points[] = {0.0, 2.0, 6.0, 12.0};
  const std::int64_t goals[] = {0, 0, 1, 2};
  const std::int64_t assists[] = {0, 1, 0, 1};
  const std::uint8_t cs[] = {0, 0, 1, 1};
  double xpts = 0, xpts_sd = 0, p_played = 0, p_60 = 0, p_goal = 0, p_assist = 0,
         p_cs = 0, p_haul = 0;
  FixtureSummary out;
  out.xpts = &xpts;
  out.xpts_sd = &xpts_sd;
  out.p_played = &p_played;
  out.p_60 = &p_60;
  out.p_goal = &p_goal;
  out.p_assist = &p_assist;
  out.p_cs = &p_cs;
  out.p_haul = &p_haul;
  aggregate_fixture(minutes, points, goals, assists, cs, 1, 4, &out);
  EXPECT_DOUBLE_EQ(xpts, 5.0);
  // var = (0+4+36+144)/4 - 25 = 46 - 25 = 21; sd = sqrt(21)
  EXPECT_NEAR(xpts_sd, std::sqrt(21.0), 1e-12);
  EXPECT_DOUBLE_EQ(p_played, 0.75);
  EXPECT_DOUBLE_EQ(p_60, 0.5);
  EXPECT_DOUBLE_EQ(p_goal, 0.5);
  EXPECT_DOUBLE_EQ(p_assist, 0.5);
  EXPECT_DOUBLE_EQ(p_cs, 0.5);
  EXPECT_DOUBLE_EQ(p_haul, 0.25);
}

TEST(SimulateMatch, ConservesGoalsWithInjectedUniforms) {
  RulesPOD rules;
  const std::size_t players = 4;
  const std::size_t n = 2;
  const std::uint8_t side[] = {0, 0, 1, 1};  // 2 home, 2 away
  const std::uint8_t position[] = {1, 2, 1, 3};
  const double minutes[] = {90, 90, 90, 90, 90, 90, 90, 90};  // (4,2) row-major
  const double gw[] = {1.0, 1.0, 1.0, 1.0};
  const double aw[] = {1.0, 1.0, 1.0, 1.0};
  const double other[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // (4,3)
  const std::int64_t home_g[] = {2, 1};
  const std::int64_t away_g[] = {1, 0};
  // max home goals = 2, away = 1
  const double home_scorer[] = {0.1, 0.2, 0.3, 0.4};  // (2,2)
  const double home_flag[] = {0.9, 0.9, 0.9, 0.9};
  const double home_ast[] = {0.5, 0.5, 0.5, 0.5};
  const double away_scorer[] = {0.1, 0.2};
  const double away_flag[] = {0.9, 0.9};
  const double away_ast[] = {0.5, 0.5};

  MatchUniforms u;
  u.home_scorer = home_scorer;
  u.home_assist_flag = home_flag;
  u.home_assister = home_ast;
  u.home_max_goals = 2;
  u.away_scorer = away_scorer;
  u.away_assist_flag = away_flag;
  u.away_assister = away_ast;
  u.away_max_goals = 1;

  SimParams params;
  params.own_goal_weight = 0.0;
  params.p_assist = 0.0;
  params.sample_other = false;

  std::vector<std::int64_t> goals(players * n);
  std::vector<std::int64_t> assists(players * n);
  std::vector<double> points(players * n);
  std::vector<std::uint8_t> cs(players * n);
  std::vector<std::int64_t> uh(n), ua(n);
  MatchSamples samples;
  samples.goals = goals.data();
  samples.assists = assists.data();
  samples.points = points.data();
  samples.clean_sheets = cs.data();
  samples.unattributed_home = uh.data();
  samples.unattributed_away = ua.data();

  std::vector<double> xpts(players), sd(players), pp(players), p60(players),
      pg(players), pa(players), pcs(players), ph(players);
  FixtureSummary summary;
  summary.xpts = xpts.data();
  summary.xpts_sd = sd.data();
  summary.p_played = pp.data();
  summary.p_60 = p60.data();
  summary.p_goal = pg.data();
  summary.p_assist = pa.data();
  summary.p_cs = pcs.data();
  summary.p_haul = ph.data();

  simulate_match_with_uniforms(home_g, away_g, side, position, minutes, gw, aw, other,
                               rules, params, u, nullptr, players, n, &samples,
                               &summary);

  for (std::size_t j = 0; j < n; ++j) {
    std::int64_t home_claimed = uh[j];
    std::int64_t away_claimed = ua[j];
    for (std::size_t i = 0; i < players; ++i) {
      if (side[i] == 0) {
        home_claimed += goals[i * n + j];
      } else {
        away_claimed += goals[i * n + j];
      }
    }
    EXPECT_EQ(home_claimed, home_g[j]);
    EXPECT_EQ(away_claimed, away_g[j]);
  }
  for (std::size_t i = 0; i < players; ++i) {
    EXPECT_TRUE(std::isfinite(xpts[i]));
    EXPECT_GE(pp[i], 0.0);
    EXPECT_LE(pp[i], 1.0);
  }
}

TEST(Points, MatchOtherAddsToExpectation) {
  RulesPOD rules;
  MatchOtherTables tables{};
  tables.delta_goals[1][0] = 1.0;  // DEF, 0 team goals
  const std::uint8_t position[] = {1};
  const std::uint8_t side[] = {0};
  const double minutes[] = {90.0};
  const std::int64_t z[] = {0};
  const std::int64_t home[] = {0};
  const std::int64_t away[] = {1};  // opp scored → no CS
  const double other[] = {0.0, 0.0, 0.5};
  double pts[1] = {};
  std::uint8_t cs[1] = {};
  fplt_mc::points(rules, position, side, minutes, z, z, home, away, other, 0.0, true,
                  false, nullptr, nullptr, &tables, 1, 1, pts, cs);
  // appearance long 2 + other bucket 0.5 + match delta 1.0 = 3.5
  EXPECT_DOUBLE_EQ(pts[0], 3.5);
}
