add_library(fw_workflow INTERFACE)
target_include_directories(fw_workflow INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(fw_workflow INTERFACE fw_common)
