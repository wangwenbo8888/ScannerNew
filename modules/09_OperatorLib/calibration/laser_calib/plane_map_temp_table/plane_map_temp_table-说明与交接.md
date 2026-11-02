> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# plane_map_temp_table · 平面映射温度表

## 1. 目标

- 逐温度档（cte=23.6e-6/°C、步距 0.2°C、±10°C）热膨胀缩放内外参与虚拟 T 后重跑 stereoRectify＋plane_map，批量产出各档 GPU 映射表——开扫门禁要的「激光档表」源头
- 定位：CPU 编排壳（无 pImpl），重活在内部 PlaneMapCuda；补偿只缩放 fx/fy/cx/cy 与 T（R 不动）
- 注意：flags 须显式传 0；勿与 core/common/calib_result_types.h 的 struct LaserPlaneMapTempTable（结果容器）混淆

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 逐温度档热膨胀缩放后重跑 stereoRectify＋plane_map 批量产各档 GPU 映射表：达成｜主类 `PlaneMapTempTable` 逐档（默认 25°C、cte=23.6e-6/°C、步距 0.2°C、±10°C）只缩放 fx/fy/cx/cy 与 T（R 不动）后重跑，产 PlaneMapTempEntry[]（每档映射表＋统计）；在用 07 calibcompute 多处（LaserChain/CalibComputeTypes/CalibSerialize/QualityGate/CalibComputePipeline），系开扫门禁「激光档表」源头
- CPU 编排壳定位：达成｜无 pImpl、重活在内部 PlaneMapCuda（CUDA）；扫描链消费其结果容器（struct LaserPlaneMapTempTable，传递件）；tests 在位（flags 须显式传 0；BUILD_CUDA=OFF 直接 success=false）

## 3. 问题与方法

- 〔待定〕问题：联调等标定数据产出（P0 主链堵点）｜方法：随标定数据产出联调演进｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制｜规范 §2.3
- 261102｜转四段新制：目标待人定稿；问题段自下一步迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段按存量参考卡颗粒度扩充（源：docs/存量档案/算子说明文档＋代码实测校准）｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
