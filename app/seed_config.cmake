# 仅缺失时播种出厂基线配置（标定两档）——不覆盖 exe 侧重标定后的本地档。
# 261002 修复：原 app/CMakeLists.txt 用 cmd /c "if not exist ... copy /Y ..."
# 拷贝，$<TARGET_FILE_DIR> 展开的正斜杠路径 cmd 的 copy 解析失败退 1（MSB3073，
# 全新构建目录首编必现；旧构建目录因文件已存在被 if not exist 掩盖）。
# 用法：cmake -DSRC=<仓库根> -DDEST=<exe 输出目录> -P seed_config.cmake
if(NOT DEFINED SRC OR NOT DEFINED DEST)
    message(FATAL_ERROR "seed_config.cmake 需要 -DSRC=<仓库根> -DDEST=<exe 输出目录>")
endif()

if(NOT EXISTS "${DEST}/config/calibration.json")
    file(COPY "${SRC}/config/calibration.json" DESTINATION "${DEST}/config")
    message(STATUS "seed: calibration.json -> ${DEST}/config/")
endif()
if(NOT EXISTS "${DEST}/config/laser_calib.json")
    file(COPY "${SRC}/config/laser_calib.json" DESTINATION "${DEST}/config")
    message(STATUS "seed: laser_calib.json -> ${DEST}/config/")
endif()
