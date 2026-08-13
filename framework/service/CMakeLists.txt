# Phase 4：StateMachine/SessionService/IState 已迁 modules/07。此处仅留 FaultHandler（待 Phase 6 迁 10）
add_library(fw_service STATIC
    FaultHandler.cpp
)
target_include_directories(fw_service PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(fw_service PUBLIC fw_common fw_infra base mod_session spdlog::spdlog)
