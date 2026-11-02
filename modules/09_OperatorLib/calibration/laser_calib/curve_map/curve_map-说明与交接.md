> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# curve_map · 极线曲线映射表

## 1. 目标

- 极线切曲线映射表（CMPU）生成＋运行期查询：ImplicitCurve[]×极线求交（25 条发射曲线），产出 CSR 压缩 tableBytes＋覆盖诊断
- 查表侧服务扫描链（cmtt_table_loader，被 07 ScanPipeline 调用）
- 红线：禁 R1/R2（矫正系直连）；rowStep 须从标定 JSON 继承、禁硬编码；FAIL 不出产物

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 极线切曲线映射表（CMPU）生成＋运行期查询：达成｜主类 `CurveMapGenerator`（ImplicitCurve[]×极线求交产 CSR 压缩 tableBytes＋覆盖诊断）＋`CurveMapTable`（运行期查询）；tests test_curve_map.cpp 在位
- 查表侧服务扫描链：达成｜查询侧在用——09 cmtt_table_loader.cpp include 本头，loader 被 07 ScanPipeline.cpp 调用；生成侧主仓 01/02/07/app 未见调用、疑在 factory_calib〔待核〕

## 3. 问题与方法

- 〔待定〕问题：生成侧调用方疑在 factory_calib（禁区未查证）｜方法：factory_calib 侧人工核实后回填｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制｜规范 §2.3
- 261102｜转四段新制：目标待人定稿；问题段自下一步迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段扩充：本算子无存量参考卡（生成侧调用方疑在 factory_calib），按已核实事实小幅补颗粒｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
