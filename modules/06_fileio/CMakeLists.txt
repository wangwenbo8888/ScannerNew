add_library(mod_fileio STATIC fileio.cpp)
target_link_libraries(mod_fileio PUBLIC fw_data fw_service fw_ui)
target_include_directories(mod_fileio PUBLIC ${CMAKE_SOURCE_DIR})
