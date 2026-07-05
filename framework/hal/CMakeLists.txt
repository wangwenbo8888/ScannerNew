add_library(fw_hal INTERFACE)
target_include_directories(fw_hal INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(fw_hal INTERFACE fw_common)
