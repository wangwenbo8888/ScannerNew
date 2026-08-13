# Module 07_session — 编排（Phase 4：StateMachine/SessionService/IState 自 framework/service 迁入）
add_library(mod_session STATIC
    StateMachine.cpp
    SessionService.cpp
)
target_include_directories(mod_session PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mod_session PUBLIC base spdlog::spdlog)
