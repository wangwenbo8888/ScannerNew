add_library(fw_crosscut INTERFACE)
target_include_directories(fw_crosscut INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(fw_crosscut INTERFACE fw_common)
