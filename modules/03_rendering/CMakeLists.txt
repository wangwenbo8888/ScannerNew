add_library(mod_rendering STATIC rendering.cpp)
target_link_libraries(mod_rendering PUBLIC fw_ui fw_algorithm fw_data)
target_include_directories(mod_rendering PUBLIC ${CMAKE_SOURCE_DIR})
