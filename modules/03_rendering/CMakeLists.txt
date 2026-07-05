# mod_rendering：渲染业务模块
#   - rendering.cpp：业务编排桩（Scanner::rendering，未来实现）
#   - display/：渲染组件（calib::，迁自 3DSCANNER260622/display；OSG + CUDA-GL interop）
add_library(mod_rendering STATIC
    rendering.cpp
    display/scanner_viewer.cpp
    display/laser_cloud_renderer.cpp
    display/marker_cloud_renderer.cpp
)

target_include_directories(mod_rendering PUBLIC
    ${CMAKE_SOURCE_DIR}                  # 工程根
    ${CMAKE_CURRENT_SOURCE_DIR}          # 本模块根
    ${CMAKE_CURRENT_SOURCE_DIR}/display  # display 组件互引（scanner_viewer.h → laser_cloud_renderer.h 等）
)

target_link_libraries(mod_rendering PUBLIC
    fw_ui fw_algorithm fw_data           # rendering.cpp 桩的业务层依赖（保留）
    mod_operatorlib                      # 上游：calib::LaserCloudFuseDeviceContext / MarkerCloudPoint
                                        #   其 PUBLIC include 自动传播 core/marker/... 解析路径
)

# === OSG（本模块专属依赖，不上抛全局）===
set(OSG_ROOT "C:/devlibs/osg-install" CACHE PATH "OSG installation directory")
target_include_directories(mod_rendering PUBLIC ${OSG_ROOT}/include)
target_link_libraries(mod_rendering PUBLIC
    ${OSG_ROOT}/lib/osg.lib
    ${OSG_ROOT}/lib/osgDB.lib
    ${OSG_ROOT}/lib/osgViewer.lib
    ${OSG_ROOT}/lib/osgGA.lib
    ${OSG_ROOT}/lib/osgUtil.lib
    ${OSG_ROOT}/lib/OpenThreads.lib
)

# === CUDA 展点核 + CUDA-GL interop ===
if(BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(mod_rendering PRIVATE display/point_expand_kernel.cu)
    target_include_directories(mod_rendering PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(mod_rendering PRIVATE CUDA::cudart)
    target_compile_definitions(mod_rendering PUBLIC BUILD_CUDA=1)
    set_source_files_properties(display/point_expand_kernel.cu PROPERTIES
        COMPILE_DEFINITIONS "FMT_UNICODE=0")
endif()

# === OpenGL（Windows 运行时）===
if(WIN32)
    target_link_libraries(mod_rendering PUBLIC opengl32)
endif()

if(MSVC)
    target_compile_options(mod_rendering PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
endif()

# === benchmark：验证渲染路径可跑（OSG 需交互式 GPU，手动运行）===
if(BUILD_TESTS)
    add_executable(marker_cloud_benchmark display/marker_cloud_benchmark.cpp)
    target_link_libraries(marker_cloud_benchmark PRIVATE mod_rendering)
    set_target_properties(marker_cloud_benchmark PROPERTIES
        VS_DEBUGGER_ENVIRONMENT "PATH=$<SHELL_PATH:${OSG_ROOT}/bin>;%PATH%")
    if(BUILD_CUDA)
        add_executable(display_benchmark display/display_benchmark.cpp)
        target_link_libraries(display_benchmark PRIVATE mod_rendering)
        set_target_properties(display_benchmark PROPERTIES
            VS_DEBUGGER_ENVIRONMENT "PATH=$<SHELL_PATH:${OSG_ROOT}/bin>;%PATH%")
    endif()
endif()
