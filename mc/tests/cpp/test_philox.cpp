#include <gtest/gtest.h>

#include "fplt_mc/rng.hpp"

using fplt_mc::Philox4x64;
using fplt_mc::philox_u01;

// Known-answer vectors for Philox4x64-10 (Random123 algorithm).
// Locked from this implementation so future edits stay bit-stable.
// Cross-check against vendored Random123 headers in a follow-up.
TEST(Philox, KnownAnswerZeroKeyZeroCtr) {
  const Philox4x64 rng(0ULL, 0ULL);
  const auto out = rng.generate(0ULL, 0ULL, 0ULL, 0ULL);
  EXPECT_EQ(out[0], 0x3bc7581c8a6461cdULL);
  EXPECT_EQ(out[1], 0x7ca6dd4b3145537fULL);
  EXPECT_EQ(out[2], 0x1003d087643ed90aULL);
  EXPECT_EQ(out[3], 0xa3922f64bf1519b3ULL);
}

TEST(Philox, KnownAnswerNonZero) {
  const Philox4x64 rng(0xDEADBEEFCAFEBABEULL, 0x0123456789ABCDEFULL);
  const auto out = rng.generate(1ULL, 2ULL, 3ULL, 4ULL);
  EXPECT_EQ(out[0], 0x524d2f8c9517e991ULL);
  EXPECT_EQ(out[1], 0xd1db3b94c8da3a8dULL);
  EXPECT_EQ(out[2], 0xcd24bd0125ae00edULL);
  EXPECT_EQ(out[3], 0x2cf22b957f11b78eULL);
}

TEST(Philox, U01InUnitInterval) {
  for (std::uint32_t i = 0; i < 64; ++i) {
    const double u = philox_u01(0xC0FFEEULL, 1u, 2u, i);
    EXPECT_GE(u, 0.0);
    EXPECT_LT(u, 1.0);
  }
}

TEST(Philox, SameInputsSameOutput) {
  const double a = philox_u01(42ULL, 3u, 7u, 11u);
  const double b = philox_u01(42ULL, 3u, 7u, 11u);
  EXPECT_EQ(a, b);
}

TEST(Philox, DifferentSeedDiffers) {
  const double a = philox_u01(1ULL, 0u, 0u, 0u);
  const double b = philox_u01(2ULL, 0u, 0u, 0u);
  EXPECT_NE(a, b);
}
