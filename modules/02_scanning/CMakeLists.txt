add_library(mod_scanning STATIC scanning.cpp)
target_link_libraries(mod_scanning PUBLIC fw_workflow fw_algorithm fw_data fw_infra)
target_include_directories(mod_scanning PUBLIC ${CMAKE_SOURCE_DIR})
