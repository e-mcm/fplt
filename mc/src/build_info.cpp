#include "fplt_mc/build_info.hpp"

#include "fplt_mc/build_info_config.hpp"

#include <string>

namespace fplt_mc {

std::unordered_map<std::string, std::string> build_info() {
  return {
      {"version", FPLT_MC_VERSION_STR},
      {"git_sha", FPLT_MC_GIT_SHA_STR},
      {"kernel_api", std::to_string(KERNEL_API)},
      {"compiler",
       std::string(FPLT_MC_COMPILER_ID) + " " + FPLT_MC_COMPILER_VERSION},
      {"flags", "-O3 -ffp-contract=off (no -ffast-math, no -march=native)"},
      {"rng", "philox4x64-10"},
      {"math", "core-math-stub"},
  };
}

}  // namespace fplt_mc
