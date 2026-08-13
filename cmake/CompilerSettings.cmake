# ============================================================================
# CompilerSettings.cmake — 编译器统一配置
# ============================================================================

if(MSVC)
    add_compile_options(/W4 /MP)
    
    # Release CRT（与 OpenCV ABI 兼容）
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    
    # Debug 映射 OpenCV Release
    foreach(_opencv_lib ${OpenCV_LIBRARIES})
        if(TARGET ${_opencv_lib})
            set_target_properties(${_opencv_lib} PROPERTIES
                MAP_IMPORTED_CONFIG_DEBUG "Release"
            )
        endif()
    endforeach()
    
    # Debug 清理
    string(REPLACE "/MDd" "/MD" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/MDd" "/MD" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/RTC1" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    
    if(MSVC_VERSION VERSION_GREATER_EQUAL "1930")
        add_compile_options(/arch:SSE2)
    endif()
endif()
