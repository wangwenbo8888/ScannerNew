# Module 10_observability — 可观测性（Phase 6：FaultHandler 自 framework/service 迁入；StateMachine/IState 自 07_session 迁入，P6-T28）
add_library(mod_observability STATIC
    FaultHandler.cpp
    StateMachine.cpp
)
target_include_directories(mod_observability PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mod_observability PUBLIC base spdlog::spdlog)
