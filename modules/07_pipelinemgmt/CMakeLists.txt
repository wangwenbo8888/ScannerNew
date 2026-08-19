# Module 07_pipelinemgmt — 流水线管理（调度底座 + 五流水线对象）
add_library(mod_pipelinemgmt STATIC)
target_include_directories(mod_pipelinemgmt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mod_pipelinemgmt PUBLIC base mod_fileio mod_operatorlib
    spdlog::spdlog ${OpenCV_LIBS})
if(BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(mod_pipelinemgmt PUBLIC CUDA::cudart)
    target_compile_definitions(mod_pipelinemgmt PUBLIC JMW_BUILD_CUDA=1)
endif()
