#pragma once

#include <array>
#include <cstdint>

namespace fplt_mc {

/// Philox4x64-10 counter-based RNG (Random123 algorithm, BSD-licensed).
///
/// Every uniform is a pure function of (key, counter). Streams are keyed by
/// (kernel_seed, stream_id); sims and slots advance the counter so output does
/// not depend on loop order or threading.
class Philox4x64 {
 public:
  static constexpr int rounds = 10;

  /// Construct with a 128-bit key (two u64 words).
  constexpr Philox4x64(std::uint64_t key0, std::uint64_t key1) noexcept
      : key0_(key0), key1_(key1) {}

  /// Construct from a single kernel seed (key1 = 0).
  explicit constexpr Philox4x64(std::uint64_t kernel_seed) noexcept
      : Philox4x64(kernel_seed, 0) {}

  /// Run one Philox4x64-10 round block on a 4×u64 counter.
  /// Returns four independent u64 outputs.
  [[nodiscard]] std::array<std::uint64_t, 4> generate(
      std::uint64_t c0, std::uint64_t c1, std::uint64_t c2,
      std::uint64_t c3) const noexcept;

  /// Uniform double in [0, 1) from the high bits of a u64
  /// (`(x >> 11) * 2^-53`).
  [[nodiscard]] static constexpr double u01(std::uint64_t x) noexcept {
    return static_cast<double>(x >> 11) * 0x1.0p-53;
  }

 private:
  std::uint64_t key0_;
  std::uint64_t key1_;
};

/// Draw one uniform in [0, 1) for (kernel_seed, stream, sim, slot).
[[nodiscard]] double philox_u01(std::uint64_t kernel_seed, std::uint32_t stream,
                                std::uint32_t sim, std::uint32_t slot) noexcept;

}  // namespace fplt_mc
