# speed_optimize — 全链速度优化沙箱（steger_dedup_replay 完整功能副本）
#
# 自包含构建（同 factory_calib 模式），数据全引用不拷：
#   - 算子源默认取原 09 树（只读）；ops/ 下放改动算子副本（按相对路径遮蔽原件）
#   - src/ = harness 链装配（复制自 steger_dedup_replay）
#   - 产物：speed_replay（CLI）＋ test_speed_replay（gtest）
#
# 构建（Release）：
#   cmake -S modules/09_operatorlib/TEST/speed_optimize -B build-rel/so `
#         -G "Visual Studio 17 2022" -A x64 -DCMAKE_CONFIGURATION_TYPES=Release `
#         -DJMW_GH_MIRROR=https://ghfast.top/
#   cmake --build build-rel/so --config Release
#   $env:PATH = 'C:\opencv-cuda-4.13.0\x64\vc17\bin;' + $env:PATH
#   ctest --test-dir build-rel/so -C Release --output-on-failure

cmake_minimum_required(VERSION 3.24)
project(speed_optimize VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# —— 第三方路径（默认同主工程；可 -D 覆盖）——
set(OpenCV_DIR "C:/opencv-cuda-4.13.0" CACHE PATH "OpenCV (Release 全组件)")
set(Eigen3_DIR "C:/devlibs/eigen-3.4.1-install/share/eigen3/cmake" CACHE PATH "Eigen3")
set(JMW_GH_MIRROR "" CACHE STRING "GitHub 镜像前缀（空=直连；本机被封时设 https://ghfast.top/）")

# 工程根（speed_optimize → TEST → 09_operatorlib → modules → 根，上溯 4 级）
get_filename_component(JMW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(OP09 "${JMW_ROOT}/modules/09_operatorlib" CACHE PATH "算子库原树（只读）")
message(STATUS "speed_optimize: JMW_ROOT=${JMW_ROOT}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(speed_common)

# —— CUDA（对齐主工程根 CMakeLists 旗标）——
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"
    CACHE FILEPATH "nvcc (pin v12.6)")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_DEPENDENCIES_USE_COMPILER OFF)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES "75;86;87" CACHE STRING "sm_75/86/87" FORCE)
if(MSVC)
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/utf-8")
endif()
string(APPEND CMAKE_CUDA_FLAGS " --extended-lambda -DFMT_UNICODE=0")

# —— 三方（FetchContent，镜像可配）——
include(FetchContent)
# 261024 产线集成守卫：BUILD_SCAN_REPLAY 挂进根 CMake 联建时，主库已建
# spdlog/json/gtest 同名 target——先查目标再拉，防 target 重定义
if(NOT TARGET spdlog::spdlog)
    find_package(spdlog 1.15 QUIET)
    if(NOT spdlog_FOUND)
        FetchContent_Declare(spdlog URL ${JMW_GH_MIRROR}https://github.com/gabime/spdlog/archive/refs/tags/v1.15.1.tar.gz)
        FetchContent_MakeAvailable(spdlog)
    endif()
endif()
if(NOT TARGET nlohmann_json::nlohmann_json)
    find_package(nlohmann_json 3.11.3 QUIET)
    if(NOT nlohmann_json_FOUND)
        FetchContent_Declare(nlohmann_json URL ${JMW_GH_MIRROR}https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
        FetchContent_MakeAvailable(nlohmann_json)
    endif()
endif()
if(NOT TARGET GTest::gtest_main)
    FetchContent_Declare(googletest URL ${JMW_GH_MIRROR}https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()
enable_testing()

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(OpenMP QUIET)

# ============================================================================
# speed_ops — 算子库（ops/ 本地副本编译——产线算子已整树拷入本目录，
# 不引用不改动原 09 树；所有优化改动只落 ops/ 副本）
# 范围：core/ 全部 + scanning/{preprocess,laser}（fusion/global_optim 不入——
#       本链不用，免 Ceres/多余编译）
# ============================================================================
# CONFIGURE_DEPENDS：新增 .cpp/.cu 时 CMake 自动重扫（否则增量构建静默漏编
# ——2026-10-23 终审 P0；沙箱规模小，重扫成本可忽略）
file(GLOB_RECURSE SO_OP_CPP CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/preprocess/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/laser/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/fusion/laser_cloud_fuse_cuda/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/calibration/*.cpp")
file(GLOB_RECURSE SO_OP_CU CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/core/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/preprocess/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/laser/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/fusion/laser_cloud_fuse_cuda/*.cu")

set(SO_SOURCES "")
foreach(f ${SO_OP_CPP} ${SO_OP_CU})
    if(f MATCHES "/tests/")
        # 算子自测不编（沙箱内用链级 gtest）
    elseif(f MATCHES "bench_|benchmark_|example_usage")
    elseif(f MATCHES "mask_sep_cli\\.cpp$")
    else()
        list(APPEND SO_SOURCES ${f})
    endif()
endforeach()
if(NOT SO_SOURCES)
    message(FATAL_ERROR "speed_optimize: ops/ 无源文件（glob 为空）")
endif()

# 头文件目录：全部算子自目录（复刻 mod_operatorlib 跨算子 include 行为）
file(GLOB_RECURSE SO_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/core/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/preprocess/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/laser/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/fusion/laser_cloud_fuse_cuda/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/calibration/*.h")
set(SO_INC_DIRS "")
foreach(h ${SO_HEADERS})
    get_filename_component(d ${h} DIRECTORY)
    list(APPEND SO_INC_DIRS ${d})
endforeach()
list(REMOVE_DUPLICATES SO_INC_DIRS)

add_library(speed_ops STATIC ${SO_SOURCES})
target_include_directories(speed_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/ops        # #include "core/common/calib_types.h"
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/core   # #include "common/calib_types.h"
    ${SO_INC_DIRS}                         # 各算子自目录（跨算子 include）
)
target_link_libraries(speed_ops PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)
if(OpenMP_FOUND)
    target_link_libraries(speed_ops PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_definitions(speed_ops PUBLIC BUILD_CUDA=1)
target_compile_definitions(speed_ops PRIVATE LM_ENABLE_TIMING=1)
if(MSVC)
    target_compile_options(speed_ops PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
endif()

# ============================================================================
# scr_runner_speed — 回放 runner 库（src 除 main.cpp 外全部；.cu 为 GPU 加速件）
# ============================================================================
file(GLOB SCR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(FILTER SCR_SOURCES EXCLUDE REGEX "main\\.cpp$")
list(FILTER SCR_SOURCES EXCLUDE REGEX "gpu_stress\\.cpp$")
file(GLOB SCR_CU "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
list(APPEND SCR_SOURCES ${SCR_CU})
if(NOT SCR_SOURCES)
    message(FATAL_ERROR "speed_optimize: src/ 无源文件（glob 为空）")
endif()
set_source_files_properties(${SCR_CU} PROPERTIES LANGUAGE CUDA)
add_library(scr_runner_speed STATIC ${SCR_SOURCES})
target_include_directories(scr_runner_speed PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${JMW_ROOT}
)
target_link_libraries(scr_runner_speed PUBLIC
    speed_ops
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_cudaarithm
    opencv_cudafilters
)
target_compile_definitions(scr_runner_speed PUBLIC
    JMW_SOURCE_ROOT="${JMW_ROOT}"
    SCR_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    WITH_CUDA_TESTS=$<BOOL:1>
)
if(MSVC)
    target_compile_options(scr_runner_speed PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
endif()

# CLI 前端
add_executable(speed_replay src/main.cpp)
target_link_libraries(speed_replay PRIVATE scr_runner_speed)
if(MSVC)
    target_compile_options(speed_replay PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif()
so_deploy_crt(speed_replay)

# GPU 满载测试
add_executable(gpu_stress src/gpu_stress.cpp)
target_link_libraries(gpu_stress PRIVATE scr_runner_speed)
if(MSVC)
  target_compile_options(gpu_stress PRIVATE \$<\$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif()
so_deploy_crt(gpu_stress)

# gtest 前端
file(GLOB SCR_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
add_executable(test_speed_replay ${SCR_TESTS})
target_link_libraries(test_speed_replay PRIVATE scr_runner_speed GTest::gtest_main)
if(MSVC)
    target_compile_options(test_speed_replay PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif()
so_deploy_crt(test_speed_replay)
add_test(NAME test_speed_replay COMMAND test_speed_replay)
