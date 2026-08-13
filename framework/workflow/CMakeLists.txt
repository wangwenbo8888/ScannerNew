# Phase 2b：IWorkflow/Pipeline(Stage) 已迁 modules/06；此处留 WorkflowContext + 3 模块工作流（待 Phase 5 迁 app）
add_library(fw_workflow STATIC
    WorkflowContext.cpp
    # Module-specific workflows (physically in module dirs)
    ${CMAKE_SOURCE_DIR}/modules/01_calibration/CalibrationWorkflow.cpp
    ${CMAKE_SOURCE_DIR}/modules/02_scanning/ScanWorkflow.cpp
    ${CMAKE_SOURCE_DIR}/modules/04_postprocessing/PostProcessWorkflow.cpp
)
target_include_directories(fw_workflow PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/modules/01_calibration
    ${CMAKE_SOURCE_DIR}/modules/02_scanning
    ${CMAKE_SOURCE_DIR}/modules/04_postprocessing
)
target_link_libraries(fw_workflow PUBLIC
    fw_common
    fw_data
    fw_service
    fw_infra
    fw_hal
    mod_fileio
    spdlog::spdlog
    ${OpenCV_LIBS}
)
target_link_libraries(fw_workflow PRIVATE mod_operatorlib)

# workflow/tests 已删（Phase 2，dead 测试）
