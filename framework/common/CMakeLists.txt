add_library(fw_common STATIC common.cpp)
target_include_directories(fw_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
find_package(OpenCV 4.13 REQUIRED COMPONENTS core)
target_link_libraries(fw_common PUBLIC ${OpenCV_LIBS})
