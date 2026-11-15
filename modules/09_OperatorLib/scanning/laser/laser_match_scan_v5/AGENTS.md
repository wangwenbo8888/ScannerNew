# laser_match_scan_v5 AGENTS — 激光线匹配 v5（人工8步后段）

> 先读：本目录 [`laser_match_scan_v5-说明与交接.md`](laser_match_scan_v5-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`LaserMatchScanSplitCpuV5`：v3 前段产物的 CPU 后段——人工8步骨架（A 门→B 门级联→D×B 裁决→回环），消费 front.cardRes 带符号残差；261115 自 TEST/fullgpu_optimize 沙盒回迁。

## 本目录禁令

- 前段必须开 cardRes 发射（`SetEmitCardRes(true)` 或 env `JMW_V3CARDRES=1`）——cardRes 空＝配置矛盾，Execute 直接失败（fail-safe 契约）
- 后段零查表——无 SetTempTable/SetCurrentTemperature，温表归前段持有；勿向后段塞表
- 参数 v3 子节与 V5 新键同 node 平铺（toJson/fromJson 同契约）；未验证机制（res_gate/depth_gate/res_clean）代码默认关，开须配置显式
- 深度门（补丁⑩）需 SetStereoQ 注入 4×4 Q，未注入自动关
- 通用契约/指纹红线见模块 AGENTS
