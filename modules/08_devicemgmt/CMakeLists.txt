# Module 08_devicemgmt — 设备管理（相机/MCU/硬件监控 + 许可证）

set(GALAXY_SDK_DIR "C:/Program Files/Daheng Imaging/GalaxySDK/Development/C++ SDK" CACHE PATH "Galaxy SDK 安装目录（见 环境配置汇总）")

add_library(mod_devicemgmt STATIC
    CameraControl.cpp
    MCUDriver.cpp
    HardwareMonitor.cpp
)

target_include_directories(mod_devicemgmt
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${GALAXY_SDK_DIR}/inc
        ${CMAKE_SOURCE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(mod_devicemgmt
    PUBLIC
        base
        ${OpenCV_LIBS}
        spdlog::spdlog
    PRIVATE
        "${GALAXY_SDK_DIR}/lib/x64/GxIAPICPPEx.lib"
)

# Galaxy SDK DLL 运行时路径（构建后复制）
set(GALAXY_DLL_DIR "${GALAXY_SDK_DIR}/lib/x64" CACHE PATH "Galaxy DLL dir")

# serial/ 协议子层（T5 起其余源文件陆续登记于此）
target_sources(mod_devicemgmt PRIVATE serial/McuFrame.cpp serial/FrameCodec.cpp serial/SerialPort.cpp serial/CommandChannel.cpp)

# PC 侧零件（T6 起陆续登记于此）
target_sources(mod_devicemgmt PRIVATE KeyManager.cpp)
target_sources(mod_devicemgmt PRIVATE MenuLogic.cpp)
target_sources(mod_devicemgmt PRIVATE KeySemantics.cpp)
target_sources(mod_devicemgmt PRIVATE ModeController.cpp)
target_sources(mod_devicemgmt PRIVATE ParamStore.cpp)

if(BUILD_TESTS)
    file(GLOB DEV_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
    foreach(t ${DEV_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        add_executable(${tname} ${t})
        target_link_libraries(${tname} PRIVATE mod_devicemgmt GTest::gtest_main)
        target_include_directories(${tname} PRIVATE ${CMAKE_SOURCE_DIR})
        add_test(NAME ${tname} COMMAND ${tname})
        jmw_deploy_crt(${tname})
    endforeach()
endif()
