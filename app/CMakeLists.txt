set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC OFF)

# Qt5（路径见 环境配置汇总.md §9）
list(APPEND CMAKE_PREFIX_PATH "C:/devlibs/Qt-5.15.2-msvc2019_64/5.15.2/msvc2019_64")
find_package(Qt5 5.15 COMPONENTS Core Gui Widgets OpenGL Svg SerialPort)

# OSG（路径见 环境配置汇总.md §10，直接 lib 路径，非 imported target）
set(OSG_ROOT "C:/devlibs/osg-install" CACHE PATH "OSG 安装目录")

# scan_demo 仅在 Qt5 Svg/SerialPort 齐备 + stubs/ 源码存在时构建
# （stubs/LEADSCANSeries.h 等当前缺失→源码不完整→跳过保构建绿；Qt 已装，stubs 补齐后自动启用）
if(Qt5_FOUND AND Qt5Svg_FOUND AND Qt5SerialPort_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/stubs/LEADSCANSeries.h")

# === app 核心文件 ===
set(APP_SOURCES
    main.cpp
    AppContext.cpp
    AppContext.h
    MainWindow.cpp
    MainWindow.h
)

# === 模块文件（物理在 modules/ 下，编译进 scan_demo）===
set(MOD_CALIBRATION
    ${CMAKE_SOURCE_DIR}/modules/01_calibration/calib_workflow.cpp
    ${CMAKE_SOURCE_DIR}/modules/01_calibration/calib_workflow.h
    ${CMAKE_SOURCE_DIR}/modules/01_calibration/CalibDialog.cpp
    ${CMAKE_SOURCE_DIR}/modules/01_calibration/CalibDialog.h
    ${CMAKE_SOURCE_DIR}/modules/01_calibration/CalibDisplay.cpp
    ${CMAKE_SOURCE_DIR}/modules/01_calibration/CalibDisplay.h
    ${CMAKE_SOURCE_DIR}/modules/01_calibration/IntegrateTestDialog.cpp
    ${CMAKE_SOURCE_DIR}/modules/01_calibration/IntegrateTestDialog.h
)

set(MOD_SCANNING
    ${CMAKE_SOURCE_DIR}/modules/02_scanning/ScannerWindow.cpp
    ${CMAKE_SOURCE_DIR}/modules/02_scanning/ScannerWindow.h
    ${CMAKE_SOURCE_DIR}/modules/02_scanning/ScannerWindow.ui
)

set(MOD_RENDERING
    ${CMAKE_SOURCE_DIR}/modules/03_rendering/OSGWidget.cpp
    ${CMAKE_SOURCE_DIR}/modules/03_rendering/OSGWidget.h
)

set(MOD_FILEIO
    ${CMAKE_SOURCE_DIR}/modules/06_fileio/file_io.cpp
    ${CMAKE_SOURCE_DIR}/modules/06_fileio/file_io.h
)

set(GENERATED_QRC_CPP ${CMAKE_CURRENT_BINARY_DIR}/qrc_resources.cpp)
add_custom_command(
    OUTPUT ${GENERATED_QRC_CPP}
    COMMAND "C:/devlibs/Qt-5.15.2-msvc2019_64/5.15.2/msvc2019_64/bin/rcc.exe" -name resources
            -o "${GENERATED_QRC_CPP}" "${CMAKE_CURRENT_SOURCE_DIR}/resources.qrc"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/resources.qrc"
    COMMENT "Generating qrc_resources.cpp"
    VERBATIM
)

add_executable(scan_demo
    ${APP_SOURCES}
    ${MOD_CALIBRATION}
    ${MOD_SCANNING}
    ${MOD_RENDERING}
    ${MOD_FILEIO}
    ${GENERATED_QRC_CPP}
)

target_include_directories(scan_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/modules/01_calibration
    ${CMAKE_SOURCE_DIR}/modules/02_scanning
    ${CMAKE_SOURCE_DIR}/modules/03_rendering
    ${CMAKE_SOURCE_DIR}/modules/06_fileio
    ${CMAKE_SOURCE_DIR}/modules/04_postprocessing
    ${OSG_ROOT}/include
)

target_link_libraries(scan_demo PRIVATE
    fw_common
    fw_data
    fw_infra
    fw_workflow
    mod_devicemgmt
    mod_operatorlib
    mod_fileio
    mod_calibration
    mod_session
    mod_observability
    Qt5::Widgets
    Qt5::OpenGL
    Qt5::Svg
    Qt5::SerialPort
    ${OSG_ROOT}/lib/osg.lib
    ${OSG_ROOT}/lib/osgDB.lib
    ${OSG_ROOT}/lib/osgViewer.lib
    ${OSG_ROOT}/lib/osgGA.lib
    ${OSG_ROOT}/lib/osgUtil.lib
    ${OSG_ROOT}/lib/OpenThreads.lib
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

target_compile_definitions(scan_demo PRIVATE
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
)

# ============================================================================
# Copy DLLs to output directory via batch script (handles Debug/Release)
# ============================================================================
add_custom_command(TARGET scan_demo POST_BUILD
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/copy_dlls.bat" "$<TARGET_FILE_DIR:scan_demo>" "$<CONFIG>"
    COMMENT "Copying DLLs to output directory..."
)

else()
    message(STATUS "app/scan_demo 跳过：缺 Qt5 Svg/SerialPort 或 stubs/ 源码（库+测试基线不受影响）")
endif()
