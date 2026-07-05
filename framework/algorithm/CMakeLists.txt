add_library(fw_algorithm INTERFACE)
target_include_directories(fw_algorithm INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(fw_algorithm INTERFACE fw_common)
