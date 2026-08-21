# Module 06_fileio — 数据管理（运行时容器 + Sink + 文件 IO）
# Phase 2 迁入：FrameBuffer/PointCloudBuffer/DeviceStateCache + 三 Sink（自 framework/data）
# B-T3：file_io 收库（点类型 cv::Point3f，命名空间 Scanner::data::fileio）
add_library(mod_fileio STATIC
    FrameBuffer.cpp
    PointCloudBuffer.cpp
    DeviceStateCache.cpp
    ParameterManager.cpp
    FrameEnricher.cpp
    ScanSessionData.cpp
    file_io.cpp
    CalibrationRepository.cpp
)
target_include_directories(mod_fileio PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mod_fileio PUBLIC base spdlog::spdlog ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json)

# ============================================================================
# 单测：每文件一 exe + 注册 ctest（仿 07_pipelinemgmt）
# ============================================================================
if(BUILD_TESTS)
    file(GLOB FI_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
    foreach(t ${FI_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        add_executable(${tname} ${t})
        target_link_libraries(${tname} PRIVATE mod_fileio GTest::gtest_main)
        target_include_directories(${tname} PRIVATE ${CMAKE_SOURCE_DIR})
        if(MSVC)
            target_compile_options(${tname} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
        endif()
        add_test(NAME ${tname} COMMAND ${tname})
        jmw_deploy_crt(${tname})
    endforeach()
endif()
