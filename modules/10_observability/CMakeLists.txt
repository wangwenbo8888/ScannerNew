# Module 10_observability — 可观测性（Phase 6：FaultHandler 自 framework/service 迁入；StateMachine/IState 自 07_session 迁入，P6-T28）
add_library(mod_observability STATIC
    CommandGate.cpp
    CrashHandler.cpp
    FaultHandler.cpp
    ObsLogger.cpp
    StateMachine.cpp
)
target_include_directories(mod_observability PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mod_observability PUBLIC base spdlog::spdlog dbghelp)

# ============================================================================
# 单测：每文件一 exe + 注册 ctest（仿 07_pipelinemgmt）
# ============================================================================
if(BUILD_TESTS)
    file(GLOB OBS_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
    foreach(t ${OBS_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        add_executable(${tname} ${t})
        if(tname STREQUAL test_crash_handler)
            # 自带 main（--crash-child 子进程分支须抢在 gtest 之前），不能链 gtest_main
            target_link_libraries(${tname} PRIVATE mod_observability GTest::gtest)
        else()
            target_link_libraries(${tname} PRIVATE mod_observability GTest::gtest_main)
        endif()
        target_include_directories(${tname} PRIVATE ${CMAKE_SOURCE_DIR})
        add_test(NAME ${tname} COMMAND ${tname})
        jmw_deploy_crt(${tname})
    endforeach()
endif()
