# Module 05_editing — 编辑工作流域对象（实施计划 P1 2026-09-05）
# SelectionService（三栏选择引擎）/ UndoRedoStack（双账本软删除账本）；
# EditWorkflow（会话主体）P4 落地时编入 app（对齐 01/02 工作流模式）。
add_library(mod_editing STATIC
    SelectionService.cpp
    UndoRedoStack.cpp
)
target_include_directories(mod_editing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mod_editing PUBLIC base ${OpenCV_LIBS})

# ============================================================================
# 单测：每文件一 exe + 注册 ctest（仿 06_fileio / 03_rendering）
# ============================================================================
if(BUILD_TESTS)
    file(GLOB EDITING_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
    foreach(t ${EDITING_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        add_executable(${tname} ${t})
        target_link_libraries(${tname} PRIVATE mod_editing GTest::gtest_main)
        if(MSVC)
            target_compile_options(${tname} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
        endif()
        add_test(NAME ${tname} COMMAND ${tname})
        jmw_deploy_crt(${tname})
    endforeach()
endif()
