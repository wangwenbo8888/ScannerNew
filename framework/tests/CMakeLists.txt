add_executable(test_framework_smoke test_smoke.cpp)
target_link_libraries(test_framework_smoke PRIVATE
    fw_common fw_algorithm fw_data fw_hal fw_infra
    fw_service fw_workflow fw_ui fw_crosscut
    GTest::gtest_main
)

add_test(NAME test_framework_smoke COMMAND test_framework_smoke)
