# base/ — 极薄共享内核（铁规：只装 types + EventBus + log，禁业务代码）
# 六审决定命名 base/（避开 modules/09_operatorlib/core/ 撞名）
add_library(base STATIC
    EventBus.cpp
)
target_include_directories(base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
