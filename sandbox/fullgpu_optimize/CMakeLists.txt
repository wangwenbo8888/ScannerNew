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
get_filename_component(JMW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
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
FetchContent_Declare(googletest URL ${JMW_GH_MIRROR}https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(OpenMP QUIET)

# ============================================================================
# 【全局优化接入 2026-11-16】Ceres 2.2.0（GBA 求解器）——配方逐行照抄主树根
# CMakeLists:142-161。⚠ 严禁定义 BUILD_CERES 宏（那是 pjc 的 Ceres 分支门控，
# 误定义＝pjc 臂静默换代码打穿门③）；⚠ USE_CUDA OFF 是硬要求（Ceres 自带 CUDA
# 与项目动态 cudart 冲突）。回滚提示：本块引入后若回滚源码，须清 build 缓存重配。
# ============================================================================
set(BUILD_TESTING_OLD ${BUILD_TESTING})
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
# 恢复本工程自有的 BUILD_TESTING 语义（Ceres FORCE 覆写过；沙箱 enable_testing
# 在其后调用不受影响，此处显式复位防顺序漂移）
set(BUILD_TESTING ${BUILD_TESTING_OLD} CACHE BOOL "" FORCE)

# ============================================================================
# speed_ops — 算子库（ops/ 本地副本编译——产线算子已整树拷入本目录，
# 不引用不改动原 09 树；所有优化改动只落 ops/ 副本）
# 范围：core/ 全部 + scanning/{preprocess,laser,global_optim}
#       ＋fusion/laser_cloud_fuse_cuda（【2026-11-16】global_optim 入列——
#       GBA 收尾阶段用；fusion 其余仍不入）
# ============================================================================
# CONFIGURE_DEPENDS：新增 .cpp/.cu 时 CMake 自动重扫（否则增量构建静默漏编
# ——2026-10-23 终审 P0；沙箱规模小，重扫成本可忽略）
file(GLOB_RECURSE SO_OP_CPP CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/preprocess/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/laser/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/fusion/laser_cloud_fuse_cuda/*.cpp"
    # 【备用算子 2026-11-20】TSDF+MC 抽薄壳（ZG 融合思路重构, CPU, 不接主链）
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/fusion/tsdf_shell_fuse/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/global_optim/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/calibration/*.cpp")
file(GLOB_RECURSE SO_OP_CU CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/core/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/preprocess/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/laser/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/fusion/laser_cloud_fuse_cuda/*.cu"
    # 【fcscan 平替 2026-11-08】标定链 CUDA 算子（laser_label/laser_match 的 Impl
    # 在 .cu 里）——此前无人消费故未入 glob；pjc_refit 需要。
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/calibration/*.cu")

set(SO_SOURCES "")
foreach(f ${SO_OP_CPP} ${SO_OP_CU})
    if(f MATCHES "/tests/")
        # 算子自测不编（沙箱内用链级 gtest）
    elseif(f MATCHES "bench_|benchmark_|example_usage")
    elseif(f MATCHES "mask_sep_cli\\.cpp$")
    elseif(f MATCHES "gba_dataset_runner\\.cpp$")  # 【GBA 2026-11-16】含 main()，双 main 冲突（主树 09 CMakeLists:24 同款排除）
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
    # 【备用算子 2026-11-20】同上（tsdf_shell_ply.h/tsdf_mc_tables.h 随算子目录入列）
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/fusion/tsdf_shell_fuse/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/scanning/global_optim/*.h"
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
    ceres  # 【GBA 2026-11-16】global_ba_cpu 求解器（主树 09:89 同款；不定义 BUILD_CERES 宏）
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

# ---- CUDA API 协作普查（cuda_census，JMW_CENSUS=1 运行期开关）----
find_package(CUDAToolkit QUIET)
if(TARGET CUDA::cupti)
    target_link_libraries(scr_runner_speed PUBLIC CUDA::cupti)
else()
    target_include_directories(scr_runner_speed PUBLIC
        "${CUDAToolkit_ROOT_DIR}/extras/CUPTI/include")
    target_link_libraries(scr_runner_speed PUBLIC
        "${CUDAToolkit_ROOT_DIR}/extras/CUPTI/lib64/cupti.lib")
endif()
file(GLOB JMW_CUPTI_DLL "${CUDAToolkit_ROOT_DIR}/extras/CUPTI/lib64/cupti64_*.dll")
if(NOT JMW_CUPTI_DLL)
    file(GLOB JMW_CUPTI_DLL "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/extras/CUPTI/lib64/cupti64_*.dll")
endif()
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
    target_link_options(speed_replay PRIVATE /MAP)   # 分配抓拍离线对号用（census trace）
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
    target_link_options(test_speed_replay PRIVATE /MAP)   # 【案件三】崩溃打印机栈离线符号化用
endif()
so_deploy_crt(test_speed_replay)

# 【F8 雅可比缺口修复验证 2026-11-08】PJC 算子单测目标：ops/**/tests/ 默认不参编
# （:96），此目标显式收录该测试文件。判别用例 RightTermAlongEpipolarShift 守护
# "uR 行进梯度"补丁——缺口未修时（块路径丢 uR 行＋正则行错位）该用例必挂。
add_executable(test_pjc
    ops/calibration/laser_calib/projector_joint_calib/tests/test_projector_joint_calib.cpp)
target_link_libraries(test_pjc PRIVATE speed_ops GTest::gtest_main)
if(MSVC)
    target_compile_options(test_pjc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif()
so_deploy_crt(test_pjc)
add_test(NAME test_pjc COMMAND test_pjc)

# 【补丁⑰·阶段0 261118】线连续性清洗离线仿真：只读 --match-img 残差库
# （frame_*_res.csv 列 lid,xL,yL,xR,res），与⑰算子逻辑同构（Hampel/率帽/豁免），
# 回答 go/no-go（剔点率 0.3%~3% / 幅值分布 / 按线分布 / 异常帧）。零算子改动。
add_executable(line_clean_sim tools/line_clean_sim.cpp)
so_deploy_crt(line_clean_sim)

# 【备用算子 2026-11-20】TSDF+MC 抽薄壳 CLI（ZG 融合思路重构; 全 CPU）:
# 点云 PLY → TSDF 距离场(就近者胜) → Marching Cubes 薄壳网格 + 厚度统计。
add_executable(tsdf_shell_extract tools/tsdf_shell_extract.cpp)
target_link_libraries(tsdf_shell_extract PRIVATE speed_ops)
so_deploy_crt(tsdf_shell_extract)

# 【fcscan 平替 2026-11-08】照片→多线重拟合（F8 右目残差）→CMTT 表再生＋对拍。
# 根治方案（专题-CMTT表uR随线号系统偏差 §10.1 步骤 1-4）的仓内执行体：
#   identity=生成器口径对拍 / refit=20 姿态重拟合 / regen=表再生。
# 依赖 2026-11-08 补拷的 ops/core/laser/{steger,epipolar_interp} 两算子目录。
add_executable(pjc_refit tools/pjc_refit.cpp)
target_link_libraries(pjc_refit PRIVATE speed_ops)
if(MSVC)
    target_compile_options(pjc_refit PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif()
so_deploy_crt(pjc_refit)

# 【算子耗时探针 2026-11-14】零落盘·可循环长跑计时：全量数据集 × N 遍，
# 只在内存累计每算子每帧毫秒（不建 outDir/无任何管线产物落盘；--out 才存
# 一份统计 CSV）。用法见 tools/op_time_probe.cpp 文件头。
add_executable(op_time_probe tools/op_time_probe.cpp)
target_link_libraries(op_time_probe PRIVATE scr_runner_speed)
if(MSVC)
    target_compile_options(op_time_probe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif()
so_deploy_crt(op_time_probe)

# ---- cupti DLL 部署（cuda_census 普查模式运行期需要）----
if(JMW_CUPTI_DLL)
    foreach(_t speed_replay test_speed_replay gpu_stress)
        add_custom_command(TARGET ${_t} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${JMW_CUPTI_DLL}" $<TARGET_FILE_DIR:${_t}>)
    endforeach()
endif()
add_test(NAME test_speed_replay COMMAND test_speed_replay)
