add_executable(jeammware main.cpp)
target_link_libraries(jeammware PRIVATE
    fw_common fw_ui fw_workflow fw_service fw_algorithm fw_data fw_hal fw_infra fw_crosscut
    mod_calibration mod_scanning mod_rendering mod_postprocess mod_editing
    mod_fileio mod_session mod_devicemgmt mod_operatorlib mod_observability mod_deploy
    jmw_sdk
)
