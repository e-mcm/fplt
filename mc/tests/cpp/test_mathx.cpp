#include <gtest/gtest.h>

#include <cmath>

#include "fplt_mc/mathx.hpp"

TEST(Mathx, ExpStubMatchesLibm) {
  // PR1 stub wraps std::exp; tighten to CORE-MATH KATs when vendored.
  EXPECT_DOUBLE_EQ(fplt_mc::mathx::exp(0.0), 1.0);
  EXPECT_NEAR(fplt_mc::mathx::exp(1.0), std::exp(1.0), 0.0);
  EXPECT_NEAR(fplt_mc::mathx::exp(-1.0), std::exp(-1.0), 0.0);
}

TEST(Mathx, ExpNegativeLambdaFinite) {
  // Poisson sampler uses exp(-λ) with small λ.
  for (double lam : {0.0, 0.1, 1.0, 5.0, 20.0}) {
    const double v = fplt_mc::mathx::exp(-lam);
    EXPECT_TRUE(std::isfinite(v));
    EXPECT_GT(v, 0.0);
    EXPECT_LE(v, 1.0);
  }
}
