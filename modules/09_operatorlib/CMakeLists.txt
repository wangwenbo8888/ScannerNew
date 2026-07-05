add_library(mod_operatorlib STATIC operatorlib.cpp)
target_link_libraries(mod_operatorlib PUBLIC fw_algorithm)
target_include_directories(mod_operatorlib PUBLIC ${CMAKE_SOURCE_DIR})
