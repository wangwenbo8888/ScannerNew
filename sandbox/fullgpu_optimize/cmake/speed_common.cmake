# 复刻自 factory_calib/cmake/factory_calib_common.cmake（speed_optimize 独立构建用）
set(SO_MSVC_REDIST_CRT_DIR
    "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Redist/MSVC/14.44.35112/x64/Microsoft.VC143.CRT"
    CACHE PATH "VS2022 MSVC redist CRT DLL dir")
get_filename_component(SO_MSVC_REDIST_MSVC_DIR "${SO_MSVC_REDIST_CRT_DIR}" DIRECTORY)
get_filename_component(SO_MSVC_REDIST_MSVC_DIR "${SO_MSVC_REDIST_MSVC_DIR}" DIRECTORY)
get_filename_component(SO_MSVC_REDIST_MSVC_DIR "${SO_MSVC_REDIST_MSVC_DIR}" DIRECTORY)
file(GLOB SO_MSVC_REDIST_OMP_GLOB
    "${SO_MSVC_REDIST_MSVC_DIR}/*/x64/Microsoft.VC143.OpenMP/vcomp140.dll")
set(SO_MSVC_REDIST_OMP_DLL "")
if(SO_MSVC_REDIST_OMP_GLOB)
    list(SORT SO_MSVC_REDIST_OMP_GLOB)
    list(REVERSE SO_MSVC_REDIST_OMP_GLOB)
    list(GET SO_MSVC_REDIST_OMP_GLOB 0 SO_MSVC_REDIST_OMP_DLL)
endif()
function(so_deploy_crt target)
    if(NOT EXISTS "${SO_MSVC_REDIST_CRT_DIR}/msvcp140.dll")
        message(WARNING "so_deploy_crt: redist CRT 目录未找到: ${SO_MSVC_REDIST_CRT_DIR} (跳过)")
        return()
    endif()
    set(SO_DEPLOY_DLLS
        "${SO_MSVC_REDIST_CRT_DIR}/msvcp140.dll"
        "${SO_MSVC_REDIST_CRT_DIR}/vcruntime140.dll"
        "${SO_MSVC_REDIST_CRT_DIR}/vcruntime140_1.dll")
    if(SO_MSVC_REDIST_OMP_DLL AND EXISTS "${SO_MSVC_REDIST_OMP_DLL}")
        list(APPEND SO_DEPLOY_DLLS "${SO_MSVC_REDIST_OMP_DLL}")
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${SO_DEPLOY_DLLS}
            "$<TARGET_FILE_DIR:${target}>"
        COMMENT "部署 VS2022 CRT DLL 到 ${target}")
endfunction()
