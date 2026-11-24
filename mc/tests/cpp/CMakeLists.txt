add_executable(fplt_mc_l0
  test_philox.cpp
  test_mathx.cpp
  test_allocate.cpp
  test_points.cpp
  test_simulate.cpp
)
target_link_libraries(fplt_mc_l0 PRIVATE fplt_mc::core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(fplt_mc_l0)
