> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# mask_extract · 激光掩膜提取（旧版）

## 1. 目标

- 功能：灰度图阈值＋形态学提激光掩膜（旧版，标定/姿态链用；扫描链已被 mask_separation 取代）
- 方法/参数：H→D 上传→GPU 阈值二值化（threshold=80）→腐蚀去噪→膨胀复形（椭圆核形态学）
- 注意：filterByArea 为空壳 TODO；高光分离路径为 261006 粘连事故产物；2 项单测（含 precision）

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 灰度图阈值＋形态学提激光掩膜（旧版，标定/姿态链用）：达成｜主类 MaskExtractCUDA：H→D→GPU 阈值二值化(threshold=80)→腐蚀(erodeSize=5)→膨胀复形；07 calibcompute/LaserChain.cpp＋posture PosturePipeline.cpp 在用；扫描链已由 mask_separation 取代（仅作 A-vs-B 对拍基线）
- 精度与高光分离路径：达成｜CPU vs CUDA 误差 <0.1px；高光分离路径（highlightErodeSize 厚度判据）为 261006-1668 粘连事故产物在册
- 面积过滤 filterByArea：未达成｜空壳 TODO（直接复制），噪声小区域可能保留
- 单测：达成｜test_mask_extract_cuda.cpp＋test_mask_extract_cuda_precision.cpp 共 2 项

## 3. 问题与方法

- 〔待定〕问题：A 模式备选前段待实测｜方法：见 scanning/preprocess/mask_separation 说明 §3.4 关联项｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制｜规范 §2.3
- 261102｜转四段新制：目标待人定稿；问题段自下一步迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段按存量参考卡颗粒度扩充（源：docs/存量档案/算子说明文档＋代码实测校准）｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
