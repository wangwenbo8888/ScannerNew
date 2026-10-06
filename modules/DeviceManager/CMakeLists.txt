# Module DeviceManager — 设备管理（相机/MCU/硬件监控 + 许可证）

set(GALAXY_SDK_DIR "C:/Program Files/Daheng Imaging/GalaxySDK/Development/C++ SDK" CACHE PATH "Galaxy SDK 安装目录（见 环境配置汇总）")

add_library(DeviceManager ${JMW_MODULE_LIB_TYPE}
    CameraControl.cpp
    CameraFactory.cpp
    MCUDriver.cpp
    HardwareMonitor.cpp
)

target_include_directories(DeviceManager
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${GALAXY_SDK_DIR}/inc
        ${CMAKE_SOURCE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# QSerialPort（CH343 兼容——Win32 WriteFile 不达固件）
find_package(Qt5 COMPONENTS SerialPort REQUIRED)

target_link_libraries(DeviceManager
    PUBLIC
        base
        ${OpenCV_LIBS}
        spdlog::spdlog
    PRIVATE
        "${GALAXY_SDK_DIR}/lib/x64/GxIAPICPPEx.lib"
        Pdh    # SelfCheckCollector CPU 占用（PDH 查询）
        Qt5::SerialPort
)

# Galaxy SDK DLL 运行时路径（构建后复制）
set(GALAXY_DLL_DIR "${GALAXY_SDK_DIR}/lib/x64" CACHE PATH "Galaxy DLL dir")

# serial/ 协议子层（T5 起其余源文件陆续登记于此）
target_sources(DeviceManager PRIVATE serial/McuFrame.cpp serial/FrameCodec.cpp serial/SerialPort.cpp serial/CommandChannel.cpp)

# PC 侧零件（T6 起陆续登记于此）
target_sources(DeviceManager PRIVATE MenuLogic.cpp)
target_sources(DeviceManager PRIVATE KeySemantics.cpp)
target_sources(DeviceManager PRIVATE ModeController.cpp)
target_sources(DeviceManager PRIVATE ParamStore.cpp)
target_sources(DeviceManager PRIVATE WarmupSequence.cpp)
target_sources(DeviceManager PRIVATE DeviceManager.cpp)
target_sources(DeviceManager PRIVATE SelfCheckCollector.cpp)

if(BUILD_TESTS)
    file(GLOB DEV_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
    foreach(t ${DEV_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        add_executable(${tname} ${t})
        # DataManager：DeviceStateCache 真件（H-T16 测试链接；测试 exe 是叶子消费方，
        # 引 06 不违「06 不链 08」铁律——库间依赖方向不变，仅叶子汇聚解析符号）
        target_link_libraries(${tname} PRIVATE DeviceManager DataManager GTest::gtest_main)
        target_include_directories(${tname} PRIVATE ${CMAKE_SOURCE_DIR})
        add_test(NAME ${tname} COMMAND ${tname})
        jmw_deploy_crt(${tname})
    endforeach()
endif()
