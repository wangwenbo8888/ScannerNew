set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC OFF)

# Qt5（路径见 环境配置汇总.md §9；JMW_QT5_PREFIX 可 -D 覆盖）
set(JMW_QT5_PREFIX "C:/Qt/Qt5.15.2/5.15.2/msvc2019_64" CACHE PATH "Qt5 安装前缀")
list(APPEND CMAKE_PREFIX_PATH "${JMW_QT5_PREFIX}")
find_package(Qt5 5.15 COMPONENTS Core Gui Widgets OpenGL Svg SerialPort)

# OSG（本机 F:/osg3.6.5/install，直接 lib 路径，非 imported target）
set(OSG_ROOT "F:/osg3.6.5/install" CACHE PATH "OSG 安装目录")

# scan_demo 仅在 Qt5 Svg/SerialPort 齐备 + stubs/ 源码存在时构建
# （stubs/ 5 个桩头已补齐，当前正常构建；若桩头缺失则跳过以保"库+测试"构建绿）
if(Qt5_FOUND AND Qt5Svg_FOUND AND Qt5SerialPort_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/stubs/LEADSCANSeries.h")

# === app 核心文件 ===
set(APP_SOURCES
    main.cpp
    AppContext.cpp
    AppContext.h
    SceneFeedAdapter.cpp
    SceneFeedAdapter.h
    MainWindow.cpp
    MainWindow.h
    WorkflowContext.cpp
    WorkflowContext.h
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
    ${CMAKE_CURRENT_SOURCE_DIR}/ScannerWindow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ScannerWindow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ScannerWindow.ui
)

# === 工作流（Phase 5：自 fw_workflow 接管编译归属；物理仍 in modules）===
set(MOD_WORKFLOWS
    ${CMAKE_SOURCE_DIR}/modules/02_scanning/ScanWorkflow.cpp
    ${CMAKE_SOURCE_DIR}/modules/02_scanning/ScanWorkflow.h
    ${CMAKE_SOURCE_DIR}/modules/01_calibration/CalibrationWorkflow.cpp
    ${CMAKE_SOURCE_DIR}/modules/01_calibration/CalibrationWorkflow.h
    ${CMAKE_SOURCE_DIR}/modules/04_postprocessing/PostProcessWorkflow.cpp
    ${CMAKE_SOURCE_DIR}/modules/04_postprocessing/PostProcessWorkflow.h
    ${CMAKE_SOURCE_DIR}/modules/05_editing/EditWorkflow.cpp
    ${CMAKE_SOURCE_DIR}/modules/05_editing/EditWorkflow.h
)

set(MOD_RENDERING
    ${CMAKE_SOURCE_DIR}/modules/03_rendering/OSGWidget.cpp
    ${CMAKE_SOURCE_DIR}/modules/03_rendering/OSGWidget.h
)

set(GENERATED_QRC_CPP ${CMAKE_CURRENT_BINARY_DIR}/qrc_resources.cpp)
add_custom_command(
    OUTPUT ${GENERATED_QRC_CPP}
    COMMAND $<TARGET_FILE:Qt5::rcc> -name resources
            -o "${GENERATED_QRC_CPP}" "${CMAKE_CURRENT_SOURCE_DIR}/resources.qrc"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/resources.qrc"
    COMMENT "Generating qrc_resources.cpp"
    VERBATIM
)

add_executable(scan_demo
    ${APP_SOURCES}
    ${MOD_CALIBRATION}
    ${MOD_SCANNING}
    ${MOD_WORKFLOWS}
    ${MOD_RENDERING}
    ${GENERATED_QRC_CPP}
)

target_include_directories(scan_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/modules/01_calibration
    ${CMAKE_SOURCE_DIR}/modules/02_scanning
    ${CMAKE_SOURCE_DIR}/modules/03_rendering
    ${CMAKE_SOURCE_DIR}/modules/05_editing
    ${CMAKE_SOURCE_DIR}/modules/06_fileio
    ${CMAKE_SOURCE_DIR}/modules/04_postprocessing
    ${OSG_ROOT}/include
)

# OSG 按配置分库（261002 修复）：Debug 链 d 后缀库（osg161-osgd.dll 等）——原写死
# Release 库名，Debug scan_demo 混链 Release OSG（/MDd exe 配 /MD DLL），CRT/迭代器
# 布局跨 DLL 不匹配，OSGWidget::initializeGL 首个 OSG 对象构造即跳 0x0 崩溃
# （0xC0000005 执行位置 0x0，261002 VS2022 Debug 实证；F: 的 OSG 带 debug 库/PDB）
foreach(_osg_lib osg osgDB osgViewer osgGA osgUtil osgText OpenThreads)
    list(APPEND APP_OSG_LIBS
        "$<$<CONFIG:Debug>:${OSG_ROOT}/lib/${_osg_lib}d.lib>"
        "$<$<NOT:$<CONFIG:Debug>>:${OSG_ROOT}/lib/${_osg_lib}.lib>")
endforeach()

target_link_libraries(scan_demo PRIVATE
    base
    mod_devicemgmt
    mod_operatorlib
    mod_fileio
    mod_pipelinemgmt
    mod_observability
    mod_editing
    Qt5::Widgets
    Qt5::OpenGL
    Qt5::Svg
    Qt5::SerialPort
    ${APP_OSG_LIBS}
    Eigen3::Eigen
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
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

# ============================================================================
# 配置播种（261002 定版）：camera.json/calibration.json/laser_calib.json 全部改为
# 「仅缺失时播种」——工厂 GUI（factory_calib）调参后把对比度直接导出到 exe 侧
# config\camera.json（扫描软件启动即读），构建不得覆盖现场调好的值。仓库根
# config/ 只作全新构建目录的出厂基线（custom_target 恒跑＝零编译增量也会补种）
# ============================================================================
add_custom_target(seed_configs ALL
    COMMAND ${CMAKE_COMMAND} "-DSRC=${CMAKE_SOURCE_DIR}" "-DDEST=$<TARGET_FILE_DIR:scan_demo>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/seed_config.cmake"
    COMMENT "Seeding config files (only if missing)..."
)

# ============================================================================
# VS2022 调试体验：调试工作目录指向 exe 目录（DLLs/platforms 旁）
# （启动项目在顶层 CMakeLists.txt 经目录属性 VS_STARTUP_PROJECT 设置）
# ============================================================================
set_target_properties(scan_demo PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>"
)

else()
    message(STATUS "app/scan_demo 跳过：缺 Qt5 Svg/SerialPort 或 stubs/ 源码（库+测试基线不受影响）")
endif()
