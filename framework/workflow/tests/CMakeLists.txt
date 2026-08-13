add_executable(test_pipeline test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE fw_common fw_data fw_workflow GTest::gtest_main)

add_test(NAME test_pipeline COMMAND test_pipeline)
