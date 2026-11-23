#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace fplt_mc {

/// Kernel API version. Bump when the C++ / Python surface changes incompatibly.
inline constexpr std::int32_t KERNEL_API = 1;

/// Return build provenance for audit.json and agent preflight checks.
[[nodiscard]] std::unordered_map<std::string, std::string> build_info();

}  // namespace fplt_mc
