# Module PipelineManager — 流水线管理（调度底座 + 流水线对象）
# 261027：C 扫描旧 lane（ScanChains/FuseConsumer/GpuPointCloudPool）退役删除——
# 扫描全链＝五链（fivechain/）。SimScanSource 保留为 app 级调试件。
set(JMW_PM_PIPELINE_SRCS pipelines/PipelineEventSink.cpp
    pipelines/calibcompute/CalibComputePipeline.cpp
    pipelines/calibcompute/CalibSerialize.cpp
    pipelines/calibcompute/CameraChain.cpp
    pipelines/calibcompute/LaserChain.cpp
    pipelines/calibcompute/QualityGate.cpp
    pipelines/posture/PostureConfirmTable.cpp
    pipelines/posture/PosturePipeline.cpp
    pipelines/postprocess/PostProcessPipeline.cpp
    pipelines/postprocess/NormalStage.cpp
    pipelines/scan/FrameObsAccumulator.cpp
    pipelines/scan/ScanPipeline.cpp
    pipelines/scan/SimScanSource.cpp   # 调试件：app 级模拟源（五链模式忽略）
    pipelines/scan/fivechain/FiveChainAdapter.cpp)  # 五链适配层
if(BUILD_CUDA)
    list(APPEND JMW_PM_PIPELINE_SRCS pipelines/scan/fivechain/fc_executor.cpp
         pipelines/scan/fivechain/gpu_accum.cu
         pipelines/scan/fivechain/cuda_census.cpp
         pipelines/scan/fivechain/ccl_cpu.cpp
         pipelines/scan/fivechain/steger_dedup_merge.cpp
         pipelines/scan/fivechain/triangle_index.cpp)  # 五链执行器族
endif()
if(BUILD_GLOBAL_OPTIM)                                                  # D 全局优化（GBA 依赖 Ceres）
    list(APPEND JMW_PM_PIPELINE_SRCS pipelines/globaloptim/GlobalOptimObject.cpp)
endif()
add_library(PipelineManager ${JMW_MODULE_LIB_TYPE} sched/FrameResultQueue.cpp sched/PCoreBroker.cpp
    sched/GpuSlotService.cpp sched/CpuTopology.cpp sched/SchedulerRuntime.cpp
    ${JMW_PM_PIPELINE_SRCS})
target_include_directories(PipelineManager PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/api)     # #include "jmw_pm_contract.h"（发布契约面）
target_link_libraries(PipelineManager PUBLIC base DataManager OperatorLib
    spdlog::spdlog ${OpenCV_LIBS})
# 261031 D1-lite 契约指纹：闭包＝自身＋base＋06＋09（ScanPipeline.h 直接
# include 06 的 SlotRing/EnhancedFrame 与 09 的 calib_result_types.h，
# 导出 ABI 由多层头共同构成）
jmw_add_contract(PipelineManager pm ${CMAKE_CURRENT_SOURCE_DIR}
    EXTRA_DIRS ${CMAKE_SOURCE_DIR}/base
              ${CMAKE_SOURCE_DIR}/modules/06_DataManager
              ${CMAKE_SOURCE_DIR}/modules/09_OperatorLib)
if(BUILD_CUDA)
    # CUDAToolkit v12.6 pin/守卫已上移根 CMakeLists（enable_language 后、任何消费方之前，
    # 原 P1-T1 模块内块于 T31 清偿）；此处仅消费其缓存结果（REQUIRED 保证未找到即失败）
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(PipelineManager PUBLIC CUDA::cudart)
    target_compile_definitions(PipelineManager PUBLIC JMW_BUILD_CUDA=1)
    # 261103 集成重放：CUDA API 协作探查（cuda_census——JMW_CENSUS=1 运行时
    # 开关；CUPTI 头/库随 CUDA 12.6 extras，与沙盒 fullgpu 同法接线；运行时
    # cupti64_*.dll 随构建拷至 bin/<配置>——缺则测试/运行 0xc0000135）
    if(TARGET CUDA::cupti)
        target_link_libraries(PipelineManager PUBLIC CUDA::cupti)
    else()
        target_include_directories(PipelineManager PUBLIC
            "${CUDAToolkit_ROOT_DIR}/extras/CUPTI/include")
        target_link_libraries(PipelineManager PUBLIC
            "${CUDAToolkit_ROOT_DIR}/extras/CUPTI/lib64/cupti.lib")
    endif()
    file(GLOB JMW_CUPTI_DLL "${CUDAToolkit_ROOT_DIR}/extras/CUPTI/lib64/cupti64_*.dll")
    if(NOT JMW_CUPTI_DLL)
        file(GLOB JMW_CUPTI_DLL
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/extras/CUPTI/lib64/cupti64_*.dll")
    endif()
    if(JMW_CUPTI_DLL)
        add_custom_command(TARGET PipelineManager POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${JMW_CUPTI_DLL}" "$<TARGET_FILE_DIR:PipelineManager>"
            COMMENT "拷 CUPTI 运行时 DLL")
    endif()
endif()

# ============================================================================
# 单测：每文件一 exe + 注册 ctest（仿 OperatorLib）
# ============================================================================
if(BUILD_TESTS)
    file(GLOB PM_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
    if(NOT BUILD_GLOBAL_OPTIM)                        # D 对象测试依赖 GBA（Ceres）
        list(REMOVE_ITEM PM_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_globaloptim.cpp")
    endif()
    foreach(t ${PM_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        add_executable(${tname} ${t})
        target_link_libraries(${tname} PRIVATE PipelineManager GTest::gtest_main)
        target_include_directories(${tname} PRIVATE ${CMAKE_SOURCE_DIR})
        if(MSVC)
            target_compile_options(${tname} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
        endif()
        add_test(NAME ${tname} COMMAND ${tname})
        jmw_deploy_crt(${tname})
    endforeach()
endif()
