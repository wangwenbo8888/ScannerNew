add_library(fw_service INTERFACE)
target_include_directories(fw_service INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(fw_service INTERFACE fw_common)
