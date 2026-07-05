add_library(fw_infra INTERFACE)
target_include_directories(fw_infra INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(fw_infra INTERFACE fw_common)
