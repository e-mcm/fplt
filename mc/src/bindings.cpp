#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>

#include "fplt_mc/build_info.hpp"
#include "fplt_mc/mathx.hpp"
#include "fplt_mc/rng.hpp"

namespace nb = nanobind;

NB_MODULE(_core, m) {
  m.doc() = "fplt-mc native Monte Carlo kernel (K2 allocate in _testing)";
  m.attr("KERNEL_API") = fplt_mc::KERNEL_API;
  m.def("build_info", &fplt_mc::build_info,
        "Return version, git SHA, compiler, and flags.");
  m.def(
      "philox_u01",
      [](std::uint64_t seed, std::uint32_t stream, std::uint32_t sim,
         std::uint32_t slot) { return fplt_mc::philox_u01(seed, stream, sim, slot); },
      nb::arg("seed"), nb::arg("stream"), nb::arg("sim"), nb::arg("slot"),
      "Draw one Philox uniform in [0, 1).");
  m.def(
      "mathx_exp", [](double x) { return fplt_mc::mathx::exp(x); }, nb::arg("x"),
      "Portable exp (CORE-MATH stub in PR1).");
}
