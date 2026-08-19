# Module 06_fileio — 数据管理（运行时容器 + Sink + 文件 IO）
# Phase 2 迁入：FrameBuffer/PointCloudBuffer/DeviceStateCache + 三 Sink（自 framework/data）
add_library(mod_fileio STATIC
    FrameBuffer.cpp
    PointCloudBuffer.cpp
    DeviceStateCache.cpp
    Pipeline.cpp
    ParameterManager.cpp
    FrameEnricher.cpp
)
target_include_directories(mod_fileio PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mod_fileio PUBLIC base spdlog::spdlog ${OpenCV_LIBS})
