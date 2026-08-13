add_library(fw_data STATIC
    FrameBuffer.cpp
    PointCloudBuffer.cpp
    DeviceStateCache.cpp
    CalibStore.cpp
)
target_include_directories(fw_data PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(fw_data PUBLIC fw_common spdlog::spdlog ${OpenCV_LIBS})

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
