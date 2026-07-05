add_library(mod_calibration STATIC calibration.cpp)
target_link_libraries(mod_calibration PUBLIC fw_workflow fw_algorithm fw_data)
target_include_directories(mod_calibration PUBLIC ${CMAKE_SOURCE_DIR})
