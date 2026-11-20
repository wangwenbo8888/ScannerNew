# cmtt_cuda AGENTS — CMTT CUDA 生成引擎

> 先读：本目录 [`cmtt_cuda-说明与交接.md`](cmtt_cuda-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)。

## 一行速记

`CmttCudaGenerator`：61 档（或非对称 N 档）CMTT sidecar 的 CUDA 编排主引擎（单档求交在 curve_map_cuda），CPU 备用＝curve_map_temp_table。

## 本目录禁令

- CUDA 与 CPU 产物等价性边界：结构一致＋真实参考档逐条目一致，**不承诺全档逐字节相同**（浮点边界）——改 kernel 须重跑 CPU↔CUDA 对拍用例
- 真数据全尺度对拍用例（E: 路径）按 fc/09 分工在 fc 仓执行，09 侧 GTEST_SKIP——勿在 09 侧引入 fc calib_io 依赖
- 通用契约/指纹红线见模块 AGENTS
