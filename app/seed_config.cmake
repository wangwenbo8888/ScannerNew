# 配置播种（仅缺失时，不覆盖现场档）——261002 定版：
#   camera.json        装机口径基线（含对比度 contrastLeft/Right）——工厂标定 GUI
#                      调好的值经「导出→拷贝到扫描机 config\」落地，构建不得覆盖
#   calibration.json / laser_calib.json  出厂标定基线——exe 侧重标定后的本地档优先
# 背景：原 POST_BUILD 里 camera.json 用 copy_if_different 每次覆盖——工厂 GUI
# 导出/人工放置的值被重编冲掉（同机与跨机两种交付都被破坏）。
# 用法：cmake -DSRC=<仓库根> -DDEST=<exe 输出目录> -P seed_config.cmake
if(NOT DEFINED SRC OR NOT DEFINED DEST)
    message(FATAL_ERROR "seed_config.cmake 需要 -DSRC=<仓库根> -DDEST=<exe 输出目录>")
endif()

if(NOT EXISTS "${DEST}/config/camera.json")
    file(COPY "${SRC}/config/camera.json" DESTINATION "${DEST}/config")
    message(STATUS "seed: camera.json -> ${DEST}/config/")
endif()
if(NOT EXISTS "${DEST}/config/calibration.json")
    file(COPY "${SRC}/config/calibration.json" DESTINATION "${DEST}/config")
    message(STATUS "seed: calibration.json -> ${DEST}/config/")
endif()
if(NOT EXISTS "${DEST}/config/laser_calib.json")
    file(COPY "${SRC}/config/laser_calib.json" DESTINATION "${DEST}/config")
    message(STATUS "seed: laser_calib.json -> ${DEST}/config/")
endif()
