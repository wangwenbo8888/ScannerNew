# curve_map_cuda AGENTS — 单档 GPU 曲线求交

> 先读：模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；同目录上位件 [`../cmtt_cuda/AGENTS.md`](../cmtt_cuda/AGENTS.md)。

## 一行速记

`CurveMapCuda`：单温度档 curve_map 的 GPU 求交生成器（cmtt_cuda 的单档内核）；`-fmad=false` 专用编译保与 CPU 版等价。

## 本目录禁令

- **-fmad=false 是等价性红线**（09 CMakeLists 按文件专用旗标）——去掉即破坏 CPU↔CUDA 对拍，禁动
- 两件形态（.h＋.cu）系 fc 原生结构豁免（09 五件套惯例不适用于本件），已记录
- 通用契约/指纹红线见模块 AGENTS
