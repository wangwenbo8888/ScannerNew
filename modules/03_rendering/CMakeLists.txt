# Module 03_rendering — INTERFACE（OSGWidget 源编入 scan_demo；本目标承载纯函数件）
# P1 渲染加固（docs/plans/2026-08-25-渲染模块加固计划.md）：
#   RenderSanity.h 纯函数挂本目标（无 Qt/OSG 依赖，测试可无头跑）
add_library(mod_rendering INTERFACE)
target_include_directories(mod_rendering INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mod_rendering INTERFACE ${OpenCV_LIBS})

# ============================================================================
# 单测：每文件一 exe + 注册 ctest（仿 06_fileio）
# ============================================================================
if(BUILD_TESTS)
    file(GLOB RENDER_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
    foreach(t ${RENDER_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        add_executable(${tname} ${t})
        target_link_libraries(${tname} PRIVATE mod_rendering GTest::gtest_main)
        if(MSVC)
            target_compile_options(${tname} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
        endif()
        add_test(NAME ${tname} COMMAND ${tname})
        jmw_deploy_crt(${tname})
    endforeach()
endif()
