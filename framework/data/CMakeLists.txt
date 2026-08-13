# Phase 2：容器已迁 modules/06（mod_fileio）；此处仅留 CalibStore（待 2b 迁 01）
add_library(fw_data STATIC
    CalibStore.cpp
)
target_include_directories(fw_data PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(fw_data PUBLIC fw_common spdlog::spdlog ${OpenCV_LIBS})
