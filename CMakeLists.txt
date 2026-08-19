cmake_minimum_required(VERSION 3.24)
project(JEAMMWARE VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
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
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"
        CACHE FILEPATH "nvcc (pin to v12.6, 避免与并存 v12.4 串味)")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_DEPENDENCIES_USE_COMPILER OFF)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "75;86;87" CACHE STRING "sm_75/86/87" FORCE)
    if(MSVC)
        string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/utf-8")
    endif()
    string(APPEND CMAKE_CUDA_FLAGS " --extended-lambda -DFMT_UNICODE=0")
endif()

# --- CUDAToolkit pin 到工程锁定的 CUDA（v12.6），防并存 v12.4 串味 ---
# 背景：FindCUDAToolkit 优先采信 CUDA 语言检测结果（CMAKE_CUDA_COMPILER_TOOLKIT_ROOT），
# 无视 CUDAToolkit_ROOT；VS 生成器下 CMake 检测可能无视上方 nvcc pin，
# 自行从注册表解析到 v12.4 并以普通变量 CMAKE_CUDA_COMPILER 遮蔽 CACHE 中的 pin 值
#（PATH/CUDA_PATH/VS CUDA props 均只有 12.6，实际编译与 OpenCV-CUDA 都是 12.6）。
# 故用 $CACHE{} 直读上方写入的 pin 原值，反推工具链根并覆写检测变量。
#（原置 modules/07_pipelinemgmt（P1-T1）；上移至此（T31 技术债）——统一在
#   enable_language(CUDA) 之后、任何 find_package(CUDAToolkit) 之前解析一次，
#   保护全部消费方、与目录遍历序无关；模块侧仅剩消费。）
if(BUILD_CUDA)
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
endif()

# 第三方路径（CACHE，骨架桩不强制 find；算子迁入后启用）
# OpenCV 有两套：Release 预装包（C:/opencv-cuda-4.13.0，/MD CRT）与 Debug 自建包
# （C:/opencv-cuda-4.13.0-debug，/MDd CRT + debug_build_guard 符号，与工程 Debug 匹配，
#   避免 Debug↔Release CRT 混用堆破坏）。默认指向 Release；Debug 构建目录用
#   -DOpenCV_DIR=C:/opencv-cuda-4.13.0-debug/x64/vc17/lib 覆盖。
set(OpenCV_DIR "C:/opencv-cuda-4.13.0" CACHE PATH "OpenCV")
set(Eigen3_DIR "C:/devlibs/eigen-3.4.1-install/share/eigen3/cmake" CACHE PATH "Eigen3")

set(JMW_GH_MIRROR "" CACHE STRING "GitHub 镜像前缀（空=直连 github.com；本机被封时设 https://ghfast.top/）")

include(FetchContent)

find_package(spdlog 1.15 QUIET)
if(NOT spdlog_FOUND)
    FetchContent_Declare(spdlog URL ${JMW_GH_MIRROR}https://github.com/gabime/spdlog/archive/refs/tags/v1.15.1.tar.gz)
    FetchContent_MakeAvailable(spdlog)
endif()

find_package(nlohmann_json 3.11.3 QUIET)
if(NOT nlohmann_json_FOUND)
    FetchContent_Declare(nlohmann_json URL ${JMW_GH_MIRROR}https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

if(BUILD_TESTS)
    FetchContent_Declare(googletest URL ${JMW_GH_MIRROR}https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)
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
        URL ${JMW_GH_MIRROR}https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.2.0.tar.gz)
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

# base/ 共享内核（Phase 1 新增；framework/* 转发壳引用 "base/*.h"，需项目根在全局 include 路径）
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/modules/06_fileio)
include_directories(${CMAKE_SOURCE_DIR}/modules/08_devicemgmt)
include_directories(${CMAKE_SOURCE_DIR}/modules/10_observability)
add_subdirectory(base)
add_subdirectory(modules)
add_subdirectory(app)
