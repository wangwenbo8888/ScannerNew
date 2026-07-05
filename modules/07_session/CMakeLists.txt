add_library(mod_session STATIC session.cpp)
target_link_libraries(mod_session PUBLIC fw_service fw_workflow fw_ui fw_data)
target_include_directories(mod_session PUBLIC ${CMAKE_SOURCE_DIR})
