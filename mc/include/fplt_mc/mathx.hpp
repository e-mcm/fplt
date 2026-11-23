#pragma once

namespace fplt_mc::mathx {

/// Portable exponential for the Poisson sampler.
///
/// PR1 ships a CORE-MATH-shaped stub that wraps `std::exp`. A later PR
/// vendors CORE-MATH `cr_exp` so results are bit-identical across glibc and
/// macOS libm. Call sites must use this entry point, never raw `std::exp`.
[[nodiscard]] double exp(double x) noexcept;

}  // namespace fplt_mc::mathx
