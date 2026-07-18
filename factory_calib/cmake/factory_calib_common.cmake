# 复刻自主工程根 CMakeLists.txt:91-106
set(FC_MSVC_REDIST_CRT_DIR
    "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Redist/MSVC/14.44.35112/x64/Microsoft.VC143.CRT"
    CACHE PATH "VS2022 MSVC redist CRT DLL dir")
function(fc_deploy_crt target)
    if(NOT EXISTS "${FC_MSVC_REDIST_CRT_DIR}/msvcp140.dll")
        message(WARNING "fc_deploy_crt: redist CRT 目录未找到: ${FC_MSVC_REDIST_CRT_DIR} (跳过)")
        return()
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FC_MSVC_REDIST_CRT_DIR}/msvcp140.dll"
            "${FC_MSVC_REDIST_CRT_DIR}/vcruntime140.dll"
            "${FC_MSVC_REDIST_CRT_DIR}/vcruntime140_1.dll"
            "$<TARGET_FILE_DIR:${target}>"
        COMMENT "部署 VS2022 CRT DLL 到 ${target}")
endfunction()
