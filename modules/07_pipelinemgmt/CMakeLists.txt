# Module 07_pipelinemgmt — 流水线管理（调度底座 + 五流水线对象）
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
    pipelines/scan/FuseConsumer.cpp
    pipelines/scan/ScanChains.cpp
    pipelines/scan/ScanPipeline.cpp)
if(BUILD_CUDA)
    list(APPEND JMW_PM_PIPELINE_SRCS pipelines/scan/GpuPointCloudPool.cpp)  # GPU 点云块池（cv::cuda 类型）
endif()
if(BUILD_GLOBAL_OPTIM)                                                  # D 全局优化（GBA 依赖 Ceres）
    list(APPEND JMW_PM_PIPELINE_SRCS pipelines/globaloptim/GlobalOptimObject.cpp)
endif()
add_library(mod_pipelinemgmt STATIC sched/FrameResultQueue.cpp sched/PCoreBroker.cpp
    sched/GpuSlotService.cpp sched/CpuTopology.cpp sched/SchedulerRuntime.cpp
    ${JMW_PM_PIPELINE_SRCS})
target_include_directories(mod_pipelinemgmt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mod_pipelinemgmt PUBLIC base mod_fileio mod_operatorlib
    spdlog::spdlog ${OpenCV_LIBS})
if(BUILD_CUDA)
    # CUDAToolkit v12.6 pin/守卫已上移根 CMakeLists（enable_language 后、任何消费方之前，
    # 原 P1-T1 模块内块于 T31 清偿）；此处仅消费其缓存结果（REQUIRED 保证未找到即失败）
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(mod_pipelinemgmt PUBLIC CUDA::cudart)
    target_compile_definitions(mod_pipelinemgmt PUBLIC JMW_BUILD_CUDA=1)
endif()

# ============================================================================
# 单测：每文件一 exe + 注册 ctest（仿 09_operatorlib）
# ============================================================================
if(BUILD_TESTS)
    file(GLOB PM_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
    if(NOT BUILD_GLOBAL_OPTIM)                        # D 对象测试依赖 GBA（Ceres）
        list(REMOVE_ITEM PM_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_globaloptim.cpp")
    endif()
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
