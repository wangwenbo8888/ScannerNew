add_library(fw_data INTERFACE)
target_include_directories(fw_data INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(fw_data INTERFACE fw_common)
