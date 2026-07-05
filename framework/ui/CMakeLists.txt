add_library(fw_ui INTERFACE)
target_include_directories(fw_ui INTERFACE ${CMAKE_SOURCE_DIR})
# fw_common 链给需要的层
target_link_libraries(fw_ui INTERFACE fw_common)
