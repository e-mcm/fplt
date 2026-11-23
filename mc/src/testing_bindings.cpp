#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include "fplt_mc/rng.hpp"

namespace nb = nanobind;

NB_MODULE(_testing, m) {
  m.doc() =
      "fplt-mc testing hooks (injected uniforms land in later PRs; Philox "
      "helpers for L0/L3).";

  m.def(
      "philox_u01_batch",
      [](std::uint64_t seed, std::uint32_t stream, std::uint32_t sim, std::uint32_t n) {
        std::vector<double> out(n);
        for (std::uint32_t i = 0; i < n; ++i) {
          out[i] = fplt_mc::philox_u01(seed, stream, sim, i);
        }
        return out;
      },
      nb::arg("seed"), nb::arg("stream"), nb::arg("sim"), nb::arg("n"),
      "Return n Philox uniforms for a fixed (seed, stream, sim).");
}
