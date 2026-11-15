> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# laser_match_scan_v5 · 激光线匹配 v5（人工8步 CPU 后段）

## 1. 目标

- 功能：激光线匹配 CPU 后段——人工 8 步骨架（支持度 A 门备选→票数比 B 门级联消元→D×B 误差裁决→裁决后回环级联）＋补丁①~⑮
- 方法/参数：消费 v3 前段产物 `LaserMatchSplitFrontV3`（复用发卡零改动，唯一新依赖＝`front.cardRes` 带符号残差列）；骨架四参数（share_fast 0.8/margin_min 2.0/quality_e_min 1.9/votes_floor 4）＋补丁键；未验证机制默认关（D7）
- 注意点：cardRes 空＝配置矛盾 Execute 失败（fail-safe）；后段零查表（无温表注入）；纯 CPU 无 CUDA 依赖

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 261115 自 TEST/fullgpu_optimize 沙盒回迁产线：达成｜源＝沙盒定稿版（261114 定稿③，金样 11,237,093 对/397 满线逐位一致＋23 单测＋ctest＋ABBA 五道门禁全绿）；主类 `LaserMatchScanSplitCpuV5`（纯 CPU 后段）
- 配套依赖已同批落地：达成｜v3 前段 cardRes 发射通道（`SetEmitCardRes`，JMW_V3CARDRES）＋v3_split 转发＋`LaserMatchSplitFrontV3.cardRes` 字段——产线 byrow CSR 缓存保留未动
- 单测：达成｜tests/test_laser_match_scan_v5.cpp＝23 用例（沙盒原版整体移植，全绿；含发射数值/fail-safe/裁决翻转/残差门族）
- 调用方接线：未达成｜07 五链尚未接 V5（沙盒口径：默认开 env JMW_LMV5、=0 退回 v4/v3、配置缺 laser_match_scan_v5 节不启用——D8 fail-safe）——本仓待人工拍板后接
- 真机联调：未达成｜交接档未决 #5（全部验证基于回放数据集）

## 3. 问题与方法

- 〔待定〕问题：07 五链接线与产线配置出口未做｜方法：按沙盒 fc_executor.cpp:334-361/harness.cpp:402-435 口径移植（JMW_LMV5 默认开＋SetEmitCardRes(true) 契约调用）｜提出 261115｜解决 —
- 〔待定〕问题：τ 系数 4.0 vs 3.0（交接档未决 #2，倾向维持 4.0）｜方法：人工拍板｜提出 261114｜解决 —
- 〔待定〕问题：#131 型错配「门范围 ≤8」选项待扫参（未决 #3）｜方法：人工扫参｜提出 261114｜解决 —

## 4. 日志

- 261115｜首次建立本说明（人工触发·AI 代笔，V5 自沙盒回迁产线当日；事实经代码实读核实）｜基线＝「现状」段｜—
- 261115｜回迁落地：v5 本体＋v3 cardRes 通道合并（byrow 保留）＋v3_split 转发＋23 单测移植，产线 ctest 4/4 绿｜—｜—
