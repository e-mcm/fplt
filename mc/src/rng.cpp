#include "fplt_mc/rng.hpp"

#include <cstddef>

namespace fplt_mc {
namespace {

// Philox4x64 multipliers and Weyl constants from Random123 (Salmon et al.).
// Algorithm copyright (c) 2010-2012, D. E. Shaw Research. BSD 3-clause.
constexpr std::uint64_t kMul0 = 0xD2B74407B1CE6E93ULL;
constexpr std::uint64_t kMul1 = 0x8E9CAF10DEFB5DAFULL;
constexpr std::uint64_t kWeyl0 = 0x9E3779B97F4A7C15ULL;  // golden ratio
constexpr std::uint64_t kWeyl1 = 0xBB67AE8584CAA73BULL;  // sqrt(3)-1? (Random123)

[[nodiscard]] inline std::uint64_t mulhilo64(std::uint64_t a, std::uint64_t b,
                                             std::uint64_t& hip) noexcept {
#if defined(__SIZEOF_INT128__)
  const __uint128_t p = static_cast<__uint128_t>(a) * b;
  hip = static_cast<std::uint64_t>(p >> 64);
  return static_cast<std::uint64_t>(p);
#else
  // Portable fallback (unused on our CI targets).
  const std::uint64_t a_lo = a & 0xffffffffULL;
  const std::uint64_t a_hi = a >> 32;
  const std::uint64_t b_lo = b & 0xffffffffULL;
  const std::uint64_t b_hi = b >> 32;
  const std::uint64_t p0 = a_lo * b_lo;
  const std::uint64_t p1 = a_lo * b_hi;
  const std::uint64_t p2 = a_hi * b_lo;
  const std::uint64_t p3 = a_hi * b_hi;
  const std::uint64_t cy =
      ((p0 >> 32) + (p1 & 0xffffffffULL) + (p2 & 0xffffffffULL)) >> 32;
  hip = p3 + (p1 >> 32) + (p2 >> 32) + cy;
  return p0 + (p1 << 32) + (p2 << 32);
#endif
}

[[nodiscard]] inline std::array<std::uint64_t, 4> round(
    std::array<std::uint64_t, 4> ctr, std::uint64_t k0, std::uint64_t k1) noexcept {
  std::uint64_t hi0 = 0;
  std::uint64_t hi1 = 0;
  const std::uint64_t lo0 = mulhilo64(kMul0, ctr[0], hi0);
  const std::uint64_t lo1 = mulhilo64(kMul1, ctr[2], hi1);
  return {hi1 ^ ctr[1] ^ k0, lo1, hi0 ^ ctr[3] ^ k1, lo0};
}

}  // namespace

std::array<std::uint64_t, 4> Philox4x64::generate(std::uint64_t c0, std::uint64_t c1,
                                                  std::uint64_t c2,
                                                  std::uint64_t c3) const noexcept {
  std::array<std::uint64_t, 4> ctr{c0, c1, c2, c3};
  std::uint64_t k0 = key0_;
  std::uint64_t k1 = key1_;
  for (int i = 0; i < rounds; ++i) {
    ctr = round(ctr, k0, k1);
    k0 += kWeyl0;
    k1 += kWeyl1;
  }
  return ctr;
}

double philox_u01(std::uint64_t kernel_seed, std::uint32_t stream, std::uint32_t sim,
                  std::uint32_t slot) noexcept {
  // Pack (stream, sim, slot) into the 256-bit counter. Four outputs per block;
  // use output[slot % 4] and bump the high counter word by slot / 4.
  const Philox4x64 rng(kernel_seed, static_cast<std::uint64_t>(stream));
  const std::uint64_t block = static_cast<std::uint64_t>(slot) / 4ULL;
  const std::uint64_t lane = static_cast<std::uint64_t>(slot) % 4ULL;
  const auto out = rng.generate(static_cast<std::uint64_t>(sim), block, 0ULL, 0ULL);
  return Philox4x64::u01(out[static_cast<std::size_t>(lane)]);
}

}  // namespace fplt_mc
