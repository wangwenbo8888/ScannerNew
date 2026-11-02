# undistort_cuda AGENTS — 激光点集去畸变矫正

> 先读：本目录 [`undistort_cuda-说明与交接.md`](undistort_cuda-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`UndistortPointsCuda`：激光点一步到矫正系——双链共用的 05 步。

## 本目录禁令

- 内参/R/P 来自 stereo_rectify_temp_table 温度补偿表（非标定文件直读）；distCoeffs 限 4~8 元素
- 通用契约/指纹红线见模块 AGENTS
