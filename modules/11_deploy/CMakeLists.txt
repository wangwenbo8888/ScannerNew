add_library(mod_deploy STATIC deploy.cpp)
target_include_directories(mod_deploy PUBLIC ${CMAKE_SOURCE_DIR})
