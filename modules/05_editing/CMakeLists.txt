add_library(mod_editing STATIC editing.cpp)
target_link_libraries(mod_editing PUBLIC fw_ui fw_service fw_data)
target_include_directories(mod_editing PUBLIC ${CMAKE_SOURCE_DIR})
