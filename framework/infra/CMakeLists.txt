# EventBus 已迁入 base/（Phase 1）；Scheduler/Watchdog 待 Phase 7 删
add_library(fw_infra STATIC Scheduler.cpp Watchdog.cpp)
target_include_directories(fw_infra PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(fw_infra PUBLIC fw_common spdlog::spdlog base)
