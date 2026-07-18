# module2_laser: 激光标定（CUDA）
find_package(OpenCV 4.13 REQUIRED COMPONENTS
    core imgproc calib3d cudaimgproc cudaarithm cudafilters flann)
find_package(Eigen3 3.4 REQUIRED)

file(GLOB_RECURSE M2_OP_CPP "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.cpp")
file(GLOB_RECURSE M2_OP_CU  "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.cu")
file(GLOB_RECURSE M2_OP_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.h")
set(M2_SOURCES "")
set(M2_TESTS "")
foreach(f ${M2_OP_CPP} ${M2_OP_CU})
    if(f MATCHES "/tests/test_[^/]*\\.cpp$")
        list(APPEND M2_TESTS ${f})
    elseif(f MATCHES "/tests/")
        # 跳过 tests 目录内非 test_ 文件
    else()
        list(APPEND M2_SOURCES ${f})
    endif()
endforeach()

set(M2_INC_DIRS "")
foreach(h ${M2_OP_HEADERS})
    get_filename_component(d ${h} DIRECTORY)
    list(APPEND M2_INC_DIRS ${d})
endforeach()
list(REMOVE_DUPLICATES M2_INC_DIRS)

if(M2_SOURCES)
    add_library(fc2_ops STATIC ${M2_SOURCES})
    target_include_directories(fc2_ops PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/operators
        ${M2_INC_DIRS})
    target_link_libraries(fc2_ops PUBLIC
        ${OpenCV_LIBS} Eigen3::Eigen spdlog::spdlog nlohmann_json::nlohmann_json)
    target_compile_definitions(fc2_ops PUBLIC BUILD_CUDA=1)
    target_compile_definitions(fc2_ops PRIVATE LM_ENABLE_TIMING=1)
    if(MSVC)
        target_compile_options(fc2_ops PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
endif()

add_executable(laser_calib laser_calib_cli.cpp)
if(TARGET fc2_ops)
    target_link_libraries(laser_calib PRIVATE fc2_ops)
endif()
fc_deploy_crt(laser_calib)

if(FC_BUILD_TESTS)
    foreach(t ${M2_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        add_executable(${tname} ${t})
        if(TARGET fc2_ops)
            target_link_libraries(${tname} PRIVATE fc2_ops GTest::gtest_main)
            target_include_directories(${tname} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/operators ${M2_INC_DIRS})
            target_compile_definitions(${tname} PRIVATE WITH_CUDA_TESTS=1)
            if(MSVC)
                target_compile_options(${tname} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
            endif()
            add_test(NAME ${tname} COMMAND ${tname})
            fc_deploy_crt(${tname})
        endif()
    endforeach()
endif()
