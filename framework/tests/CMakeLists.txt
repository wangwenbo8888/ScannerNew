add_executable(test_framework_smoke test_smoke.cpp)
target_link_libraries(test_framework_smoke PRIVATE
    fw_common fw_ui fw_workflow fw_service fw_algorithm fw_data fw_hal fw_infra fw_crosscut
    jmw_sdk
    GTest::gtest_main
)
target_include_directories(test_framework_smoke PRIVATE ${CMAKE_SOURCE_DIR})
add_test(NAME framework_smoke COMMAND test_framework_smoke)
