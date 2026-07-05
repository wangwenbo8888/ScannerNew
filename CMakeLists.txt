cmake_minimum_required(VERSION 3.24)
project(JEAMMWARE VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

option(BUILD_CUDA "Build with CUDA support" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_UI "Build Qt UI layer (deferred)" OFF)

if(BUILD_CUDA)
    set(CMAKE_CUDA_DEPENDENCIES_USE_COMPILER OFF)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "75;86;87" CACHE STRING "sm_75/86/87" FORCE)
    if(MSVC)
        string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/utf-8")
    endif()
    string(APPEND CMAKE_CUDA_FLAGS " --extended-lambda -DFMT_UNICODE=0")
endif()

# 第三方路径（CACHE，骨架桩不强制 find；算子迁入后启用）
set(OpenCV_DIR "C:/opencv-cuda-4.13.0" CACHE PATH "OpenCV")
set(Eigen3_DIR "C:/devlibs/eigen-3.4.1-install/share/eigen3/cmake" CACHE PATH "Eigen3")

include(FetchContent)

find_package(spdlog 1.15 QUIET)
if(NOT spdlog_FOUND)
    FetchContent_Declare(spdlog URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.1.tar.gz)
    FetchContent_MakeAvailable(spdlog)
endif()

find_package(nlohmann_json 3.11.3 QUIET)
if(NOT nlohmann_json_FOUND)
    FetchContent_Declare(nlohmann_json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

if(BUILD_TESTS)
    FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# === 算子依赖（算子迁入启用）===
find_package(OpenCV 4.13 REQUIRED COMPONENTS
    core imgproc calib3d cudaimgproc cudaarithm cudafilters flann)
find_package(Eigen3 3.4 REQUIRED)

# Ceres（global_optim/GBA 用；精简配置，免 SuiteSparse，用 Eigen 稀疏）
option(BUILD_GLOBAL_OPTIM "Build global optimization (GBA + Ceres)" ON)
if(BUILD_GLOBAL_OPTIM)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCUMENTATION OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "" FORCE)
    set(GFLAGS OFF CACHE BOOL "" FORCE)
    set(MINIGLOG ON CACHE BOOL "" FORCE)
    set(SUITESPARSE OFF CACHE BOOL "" FORCE)
    set(CXSPARSE OFF CACHE BOOL "" FORCE)
    set(ACCELERATESPARSE OFF CACHE BOOL "" FORCE)
    set(EIGENSPARSE ON CACHE BOOL "" FORCE)
    set(LAPACK OFF CACHE BOOL "" FORCE)
    set(SCHUR_SPECIALIZATIONS OFF CACHE BOOL "" FORCE)
    set(USE_CUDA OFF CACHE BOOL "" FORCE)  # Ceres 自带 CUDA 与项目动态 cudart 冲突
    FetchContent_Declare(ceres
        URL https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.2.0.tar.gz)
    FetchContent_MakeAvailable(ceres)
endif()

# ----------------------------------------------------------------------------
# CRT 部署：System32 的 MSVCP140.dll 可能是旧版(会崩溃)，把 VS redist 新版
# CRT DLL 复制到 exe 旁(exe 目录 DLL 搜索最优先)，保证 OpenCV 等正常初始化。
# ----------------------------------------------------------------------------
set(JMW_MSVC_REDIST_CRT_DIR
    "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Redist/MSVC/14.44.35112/x64/Microsoft.VC143.CRT"
    CACHE PATH "VS2022 MSVC redist CRT DLL dir (含 msvcp140/vcruntime140)")
function(jmw_deploy_crt target)
    if(NOT EXISTS "${JMW_MSVC_REDIST_CRT_DIR}/msvcp140.dll")
        message(WARNING "jmw_deploy_crt: redist CRT 目录未找到: ${JMW_MSVC_REDIST_CRT_DIR} (跳过)")
        return()
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${JMW_MSVC_REDIST_CRT_DIR}/msvcp140.dll"
            "${JMW_MSVC_REDIST_CRT_DIR}/vcruntime140.dll"
            "${JMW_MSVC_REDIST_CRT_DIR}/vcruntime140_1.dll"
            "$<TARGET_FILE_DIR:${target}>"
        COMMENT "部署 VS2022 CRT DLL 到 ${target} 目录")
endfunction()

add_subdirectory(framework)
add_subdirectory(modules)
add_subdirectory(sdk)
add_subdirectory(app)
