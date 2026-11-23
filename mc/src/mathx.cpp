#include "fplt_mc/mathx.hpp"

#include <cmath>

namespace fplt_mc::mathx {

double exp(double x) noexcept {
  // Stub: replace with CORE-MATH cr_exp (vendored under third_party/core-math)
  // so exp is bit-identical on glibc and macOS libm. See native-mc-build-spec §3.
  return std::exp(x);
}

}  // namespace fplt_mc::mathx
