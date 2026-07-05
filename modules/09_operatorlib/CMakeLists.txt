# mod_operatorlib：全部算子（core/calibration/scanning/global_optim）一个库
#
# 来源：3DSCANNER 工程机械迁移（命名空间保留 calib::）
# 范围：core/{common,vision,laser,marker} + calibration/算子 + scanning/{preprocess,laser,fusion,global_optim}
# 不含：pipeline/（workflow 编排）、scheduler/（infra）、bench/benchmark

# === 收集所有源（glob + 分类：库源 / 测试 / 排除）===
file(GLOB_RECURSE OP_ALL_CPP
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/calibration/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/scanning/*.cpp")
file(GLOB_RECURSE OP_ALL_CU
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/calibration/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/scanning/*.cu")

set(OP_SOURCES "")
set(OP_TESTS "")
foreach(f ${OP_ALL_CPP} ${OP_ALL_CU})
    if(f MATCHES "/tests/test_[^/]*\\.cpp$")
        list(APPEND OP_TESTS ${f})
    elseif(f MATCHES "/tests/|bench_|benchmark_|example_usage")
        # 跳过：bench/benchmark/非 test_ 的 tests 目录内文件
    elseif(f MATCHES "gba_dataset_runner\\.cpp$")
        # 独立 exe，不入库
    else()
        list(APPEND OP_SOURCES ${f})
    endif()
endforeach()

# 收集所有含头文件的目录 → include 路径（复刻 3DSCANNER 各算子 PUBLIC include 自目录的行为，
# 使跨算子 include 如 plane_map_temp_table → plane_map_cuda.h 可解析）
file(GLOB_RECURSE OP_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/calibration/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/scanning/*.h")
set(OP_INC_DIRS "")
foreach(h ${OP_HEADERS})
    get_filename_component(d ${h} DIRECTORY)
    list(APPEND OP_INC_DIRS ${d})
endforeach()
list(REMOVE_DUPLICATES OP_INC_DIRS)

# === BUILD_GLOBAL_OPTIM 关时剔除 GBA 源（依赖 Ceres）===
if(NOT BUILD_GLOBAL_OPTIM)
    list(FILTER OP_SOURCES EXCLUDE REGEX "global_ba_cpu\\.cpp$")
endif()

# ============================================================================
add_library(mod_operatorlib STATIC ${OP_SOURCES})
target_include_directories(mod_operatorlib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}          # #include "core/common/calib_types.h"
    ${CMAKE_CURRENT_SOURCE_DIR}/core     # #include "common/calib_types.h"（3DSCANNER 风格）
    ${CMAKE_SOURCE_DIR}                  # 工程根
    ${OP_INC_DIRS}                       # 各算子自目录（跨算子 include）
)

target_link_libraries(mod_operatorlib PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

if(BUILD_CUDA)
    target_compile_definitions(mod_operatorlib PUBLIC BUILD_CUDA=1)
else()
    target_compile_definitions(mod_operatorlib PUBLIC BUILD_CUDA=0)
endif()

# laser_match_scan 内部计时开关（target 级，仅该算子源文件引用，与 3DSCANNER 一致）
target_compile_definitions(mod_operatorlib PRIVATE LM_ENABLE_TIMING=1)

# calibration/scanning 源文件体量大，MSVC 需 /bigobj
if(MSVC)
    target_compile_options(mod_operatorlib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif()

# === Ceres（GBA）===
if(BUILD_GLOBAL_OPTIM)
    target_link_libraries(mod_operatorlib PUBLIC ceres)
endif()

# === gba_dataset_runner exe（GBA 离线工具，含 main）===
if(BUILD_GLOBAL_OPTIM)
    add_executable(gba_dataset_runner
        ${CMAKE_CURRENT_SOURCE_DIR}/scanning/global_optim/gba_dataset_runner.cpp)
    target_link_libraries(gba_dataset_runner PRIVATE mod_operatorlib ceres Eigen3::Eigen)
    if(MSVC)
        target_compile_options(gba_dataset_runner PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
    endif()
    jmw_deploy_crt(gba_dataset_runner)   # 部署 VS2022 CRT DLL（OpenCV 运行时依赖）
endif()

# ============================================================================
# 算子单测：每文件一个 exe + 注册 ctest
# ============================================================================
if(BUILD_TESTS)
    foreach(t ${OP_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        add_executable(${tname} ${t})
        target_link_libraries(${tname} PRIVATE mod_operatorlib GTest::gtest_main)
        target_include_directories(${tname} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/core
            ${CMAKE_SOURCE_DIR})
        # CUDA 测试开关（与 3DSCANNER 一致；无害于纯 CPU 测试）
        target_compile_definitions(${tname} PRIVATE WITH_CUDA_TESTS=$<BOOL:${BUILD_CUDA}>)
        if(MSVC)
            target_compile_options(${tname} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
        endif()
        if(BUILD_GLOBAL_OPTIM)
            target_link_libraries(${tname} PRIVATE ceres)
        endif()
        add_test(NAME ${tname} COMMAND ${tname})
        jmw_deploy_crt(${tname})            # 部署 VS2022 CRT DLL（OpenCV 运行时依赖）
    endforeach()
endif()
