> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# curve_map_temp_table · 曲线映射温度表

## 1. 目标

- curve_map 的温度档 sidecar 批量生成（61 个温度档）：输入 ImplicitCurve[]＋两张温度表结果，产出 sidecarBytes（每档 blob 只驻此一份）＋逐档结果
- 产物 .bin 由扫描链消费（07 ScanPipeline→cmtt_table_loader 查表）
- 红线：参考档失败则整体 sidecar 为空——不得部分出产物

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 61 温度档 sidecar 批量生成：达成｜主类 `CurveMapTempTableGenerator` 吃 ImplicitCurve[]＋StereoRectifyTempTableResult＋LaserExtrinsicCompensateCPUResult，产 sidecarBytes（每档 blob 只驻此一份）＋逐档结果；5 个 tests 在位
- 产物 .bin 供扫描链查表：达成｜扫描链 07 ScanPipeline→cmtt_table_loader 消费 sidecar .bin（产物在用）；生成器本体主仓未见调用、疑 factory_calib 侧〔待核〕

## 3. 问题与方法

- 〔待定〕问题：生成侧调用方疑在 factory_calib（禁区未查证）｜方法：factory_calib 侧人工核实后回填｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制｜规范 §2.3
- 261102｜转四段新制：目标待人定稿；问题段自下一步迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段扩充：本算子无存量参考卡（生成侧调用方疑在 factory_calib），按已核实事实小幅补颗粒｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
