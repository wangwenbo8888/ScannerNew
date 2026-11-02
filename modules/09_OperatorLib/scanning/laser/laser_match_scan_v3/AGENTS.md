# laser_match_scan_v3 AGENTS — 激光线匹配 v3（整版）

> 先读：本目录 [`laser_match_scan_v3-说明与交接.md`](laser_match_scan_v3-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`LaserMatchScanCudaV3`：v3 整版——产线默认已被 split 拆分版接替，本件退为 JMW_FC_FULLGPU 实验臂。

## 本目录禁令

- GpuMat 结果是内部缓冲视图——生命周期至下次 Execute/Destroy/setParams，勿跨帧持有
- 温表注入只走 SetTempTable（cmtt_table_loader 装载）——勿旁路自读文件
- 通用契约/指纹红线见模块 AGENTS
