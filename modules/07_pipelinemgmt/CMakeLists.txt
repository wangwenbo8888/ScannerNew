# Module 07_pipelinemgmt — 流水线管理（调度底座 + 五流水线对象）
add_library(mod_pipelinemgmt STATIC sched/FrameResultQueue.cpp sched/PCoreBroker.cpp)
target_include_directories(mod_pipelinemgmt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mod_pipelinemgmt PUBLIC base mod_fileio mod_operatorlib
    spdlog::spdlog ${OpenCV_LIBS})
if(BUILD_CUDA)
    # --- CUDAToolkit pin 到工程锁定的 CUDA（v12.6），防并存 v12.4 串味 ---
    # 背景：FindCUDAToolkit 优先采信 CUDA 语言检测结果（CMAKE_CUDA_COMPILER_TOOLKIT_ROOT），
    # 无视 CUDAToolkit_ROOT；VS 生成器下 CMake 检测可能无视根 CMakeLists 的 nvcc pin，
    # 自行从注册表解析到 v12.4 并以普通变量 CMAKE_CUDA_COMPILER 遮蔽 CACHE 中的 pin 值
    #（PATH/CUDA_PATH/VS CUDA props 均只有 12.6，实际编译与 OpenCV-CUDA 都是 12.6）。
    # 故用 $CACHE{} 直读根 CMakeLists 写入的 pin 原值，反推工具链根并覆写检测变量。
    get_filename_component(_jmw_cuda_root "$CACHE{CMAKE_CUDA_COMPILER}" DIRECTORY)  # .../v12.6/bin
    get_filename_component(_jmw_cuda_root "${_jmw_cuda_root}" DIRECTORY)            # .../v12.6
    set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "${_jmw_cuda_root}")
    set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "${_jmw_cuda_root}/lib/x64")
    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "${_jmw_cuda_root}/include")
    set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "${_jmw_cuda_root}/lib/x64")
    set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "")
    if(EXISTS "${_jmw_cuda_root}/version.json")
        file(READ "${_jmw_cuda_root}/version.json" _jmw_cuda_ver_json)
        string(REGEX MATCH "\"version\"[ \t]*:[ \t]*\"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _ "${_jmw_cuda_ver_json}")
        set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "${CMAKE_MATCH_1}")
    endif()
    set(CUDAToolkit_ROOT "${_jmw_cuda_root}")
    unset(CUDAToolkit_DIR CACHE)                 # 防旧缓存残留串味
    unset(CUDAToolkit_CUPTI_INCLUDE_DIR CACHE)   # 同上：缓存命中会短路按新根的 re-search
    unset(CUDA_CUDART CACHE)                     # 同上：FindCUDAToolkit 复用 FindCUDA 时代的缓存名
    find_package(CUDAToolkit REQUIRED)
    # 守卫：CUDAToolkit 必须解析到 pin 的工具链根，不一致即失败，绝不静默混链
    cmake_path(IS_PREFIX _jmw_cuda_root "${CUDAToolkit_LIBRARY_ROOT}" _jmw_cudatk_ok NORMALIZE)
    if(NOT _jmw_cudatk_ok)
        message(FATAL_ERROR "CUDAToolkit 解析到 ${CUDAToolkit_LIBRARY_ROOT}，与 pin 的 "
            "${_jmw_cuda_root}（来自 CMAKE_CUDA_COMPILER）不一致，存在 v12.4/v12.6 串味风险")
    endif()
    unset(_jmw_cuda_root)
    unset(_jmw_cudatk_ok)
    unset(_jmw_cuda_ver_json)
    target_link_libraries(mod_pipelinemgmt PUBLIC CUDA::cudart)
    target_compile_definitions(mod_pipelinemgmt PUBLIC JMW_BUILD_CUDA=1)
endif()

# ============================================================================
# 单测：每文件一 exe + 注册 ctest（仿 09_operatorlib）
# ============================================================================
if(BUILD_TESTS)
    file(GLOB PM_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
    foreach(t ${PM_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        add_executable(${tname} ${t})
        target_link_libraries(${tname} PRIVATE mod_pipelinemgmt GTest::gtest_main)
        target_include_directories(${tname} PRIVATE ${CMAKE_SOURCE_DIR})
        if(MSVC)
            target_compile_options(${tname} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
        endif()
        add_test(NAME ${tname} COMMAND ${tname})
        jmw_deploy_crt(${tname})
    endforeach()
endif()
