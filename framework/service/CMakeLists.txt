add_library(fw_service STATIC
    StateMachine.cpp
    FaultHandler.cpp
    SessionService.cpp
)
target_include_directories(fw_service PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(fw_service PUBLIC fw_common fw_infra spdlog::spdlog)
