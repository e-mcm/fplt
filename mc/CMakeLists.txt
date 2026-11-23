cmake_minimum_required(VERSION 3.26)
project(fplt_mc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FPLT_MC_BUILD_PYTHON "Build nanobind Python extension" ON)
option(FPLT_MC_BUILD_TESTS "Build GoogleTest L0 suite" OFF)
option(FPLT_MC_SANITIZE "Enable ASan+UBSan (Debug)" OFF)

# Portable FP: no FMA contraction, no fast-math, no -march=native.
add_compile_options(-O3 -ffp-contract=off)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  add_compile_options(-march=x86-64-v2)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  add_compile_options(-march=armv8-a)
endif()

if(FPLT_MC_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

set(FPLT_MC_GIT_SHA "unknown")
execute_process(
  COMMAND git rev-parse --short HEAD
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE _git_sha
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE _git_rc
)
if(_git_rc EQUAL 0 AND NOT _git_sha STREQUAL "")
  set(FPLT_MC_GIT_SHA "${_git_sha}")
endif()
# scikit-build / PEP517 isolated builds often lack .git; allow override from CI.
if(DEFINED ENV{GITHUB_SHA} AND NOT "$ENV{GITHUB_SHA}" STREQUAL "")
  string(SUBSTRING "$ENV{GITHUB_SHA}" 0 7 FPLT_MC_GIT_SHA)
endif()

set(FPLT_MC_VERSION "${PROJECT_VERSION}")
if(NOT FPLT_MC_VERSION OR FPLT_MC_VERSION STREQUAL "")
  # Prefer the Python package version when PROJECT_VERSION is unset.
  file(READ "${CMAKE_SOURCE_DIR}/pyproject.toml" _PYPROJECT)
  string(REGEX MATCH "version = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _ ${_PYPROJECT})
  set(FPLT_MC_VERSION "${CMAKE_MATCH_1}")
endif()

configure_file(
  "${CMAKE_SOURCE_DIR}/include/fplt_mc/build_info_config.hpp.in"
  "${CMAKE_BINARY_DIR}/generated/fplt_mc/build_info_config.hpp"
  @ONLY
)

add_library(fplt_mc_core STATIC
  src/rng.cpp
  src/mathx.cpp
  src/build_info.cpp
  src/allocate.cpp
  src/points.cpp
)
add_library(fplt_mc::core ALIAS fplt_mc_core)
# Required when the static lib is linked into the nanobind shared module.
set_target_properties(fplt_mc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fplt_mc_core
  PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_BINARY_DIR}/generated"
    "${CMAKE_SOURCE_DIR}/third_party"
)
target_compile_definitions(fplt_mc_core
  PUBLIC
    FPLT_MC_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    FPLT_MC_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
)

if(FPLT_MC_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(nanobind CONFIG REQUIRED)

  # STABLE_ABI → cp312-abi3 wheels (Python ≥ 3.12); avoids rebuild at 3.14.
  nanobind_add_module(_core STABLE_ABI src/bindings.cpp)
  target_link_libraries(_core PRIVATE fplt_mc::core)
  install(TARGETS _core LIBRARY DESTINATION . COMPONENT python)

  nanobind_add_module(_testing STABLE_ABI src/testing_bindings.cpp)
  target_link_libraries(_testing PRIVATE fplt_mc::core)
  install(TARGETS _testing LIBRARY DESTINATION . COMPONENT python)
endif()

if(FPLT_MC_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(tests/cpp)
endif()
