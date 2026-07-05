add_library(mod_devicemgmt STATIC devicemgmt.cpp)
target_link_libraries(mod_devicemgmt PUBLIC fw_hal fw_service fw_ui fw_data fw_crosscut)
target_include_directories(mod_devicemgmt PUBLIC ${CMAKE_SOURCE_DIR})
