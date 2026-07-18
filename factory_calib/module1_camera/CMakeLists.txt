# module1_camera: 相机标定（纯 CPU）
# 复刻 modules/09_operatorlib/CMakeLists.txt 的 glob + include 策略，但只 CPU

find_package(OpenCV 4.13 REQUIRED COMPONENTS core imgproc calib3d flann)
find_package(Eigen3 3.4 REQUIRED)

# === 收集算子源 ===
file(GLOB_RECURSE M1_OP_CPP "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.cpp")
file(GLOB_RECURSE M1_OP_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.h")
set(M1_SOURCES "")
set(M1_TESTS "")
foreach(f ${M1_OP_CPP})
    if(f MATCHES "/tests/test_[^/]*\\.cpp$")
        list(APPEND M1_TESTS ${f})
    elseif(f MATCHES "/tests/")
        # 跳过 tests 目录内非 test_ 文件
    else()
        list(APPEND M1_SOURCES ${f})
    endif()
endforeach()

# 每个算子目录都加入 include 路径（跨算子 include）
set(M1_INC_DIRS "")
foreach(h ${M1_OP_HEADERS})
    get_filename_component(d ${h} DIRECTORY)
    list(APPEND M1_INC_DIRS ${d})
endforeach()
list(REMOVE_DUPLICATES M1_INC_DIRS)

# === 算子静态库 ===
if(M1_SOURCES)
    add_library(fc1_ops STATIC ${M1_SOURCES})
    target_include_directories(fc1_ops PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/operators   # "common/xxx.h"
        ${M1_INC_DIRS})                          # 跨算子 include
    target_link_libraries(fc1_ops PUBLIC
        ${OpenCV_LIBS} Eigen3::Eigen spdlog::spdlog nlohmann_json::nlohmann_json)
    if(MSVC)
        target_compile_options(fc1_ops PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
    endif()
endif()

# === camera_calib.exe ===
add_executable(camera_calib camera_calib_cli.cpp)
if(TARGET fc1_ops)
    target_link_libraries(camera_calib PRIVATE fc1_ops)
endif()
target_link_libraries(camera_calib PRIVATE ${OpenCV_LIBS} nlohmann_json::nlohmann_json spdlog::spdlog)
fc_deploy_crt(camera_calib)

# === 测试 ===
if(FC_BUILD_TESTS)
    foreach(t ${M1_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        add_executable(${tname} ${t})
        target_link_libraries(${tname} PRIVATE fc1_ops GTest::gtest_main)
        target_include_directories(${tname} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/operators ${M1_INC_DIRS})
        target_compile_definitions(${tname} PRIVATE WITH_CUDA_TESTS=0)
        if(MSVC)
            target_compile_options(${tname} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
        endif()
        add_test(NAME ${tname} COMMAND ${tname})
        fc_deploy_crt(${tname})
    endforeach()
    # 模块1 自有测试（Phase 2/3 添加，glob 自带捕获）
endif()
