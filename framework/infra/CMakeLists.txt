add_library(fw_infra STATIC EventBus.cpp Scheduler.cpp Watchdog.cpp)
target_include_directories(fw_infra PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(fw_infra PUBLIC fw_common spdlog::spdlog)
