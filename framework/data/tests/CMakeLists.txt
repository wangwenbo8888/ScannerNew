add_executable(test_ringbuffer test_ringbuffer.cpp)
target_link_libraries(test_ringbuffer PRIVATE fw_common GTest::gtest_main)

add_test(NAME test_ringbuffer COMMAND test_ringbuffer)
