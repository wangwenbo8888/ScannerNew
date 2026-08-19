# Module 07_session — 编排（Phase 4：SessionService 自 framework/service 迁入；StateMachine/IState 已迁 10_observability，P6-T28）
add_library(mod_session STATIC
    SessionService.cpp
)
target_include_directories(mod_session PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mod_session PUBLIC base spdlog::spdlog)
