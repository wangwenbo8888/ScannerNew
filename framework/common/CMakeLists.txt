add_library(fw_common STATIC common.cpp)
target_sources(fw_common PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/quality_flag.h ${CMAKE_CURRENT_SOURCE_DIR}/types.h)
target_include_directories(fw_common PUBLIC ${CMAKE_SOURCE_DIR})
