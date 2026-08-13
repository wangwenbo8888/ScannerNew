# PatchVcxproj.cmake — 移除 vcxproj 中 CUDA 不兼容的编译选项
file(READ "${FILE}" content)
string(REPLACE "${PATTERN}" "${REPLACE}" content "${content}")
file(WRITE "${FILE}" "${content}")
