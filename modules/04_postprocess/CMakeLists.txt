add_library(mod_postprocess STATIC postprocess.cpp)
target_link_libraries(mod_postprocess PUBLIC fw_workflow fw_algorithm fw_data)
target_include_directories(mod_postprocess PUBLIC ${CMAKE_SOURCE_DIR})
