# Module 01_calibration — 标定（Phase 2c：CalibStore 自 framework/data 迁入）
add_library(mod_calibration STATIC
    CalibStore.cpp
)
target_include_directories(mod_calibration PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mod_calibration PUBLIC base spdlog::spdlog ${OpenCV_LIBS} mod_fileio)
