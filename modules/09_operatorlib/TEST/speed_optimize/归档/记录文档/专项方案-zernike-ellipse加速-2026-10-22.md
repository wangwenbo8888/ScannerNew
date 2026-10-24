# 专项实施方案：zernike_edge＋ellipse_fit CPU 算子加速（2026-10-22 三轮九代理终版）

> 流程：三轮×三代理（深析→方案→终审）已毕，本文件为裁决合并后的执行版。
> 目标：zernike 中位 7.0→≤3.0ms（门 −50%）；ellipse 5.0→双门（无 E3 ≥−12%，含 E3 ≥−30%）；
> FPS 仅作回归护栏（±3%，GPU 链 8.8ms 是下一约束，marker 侧收益买余量与抖动收敛）。
> 边界：只改 modules/09_operatorlib/TEST/speed_optimize/；验收＝82+77 用例全绿＋
> 399 帧 golden 对拍 0 diffs（serial 0 flip＋fivechain ≤1 flip）。

## 批次表（每批：构建→ctest→双模式对拍→A/B）

| 批 | 内容 | 等价性预期 | A/B |
|---|---|---|---|
| B1 | **Z1**：cv::setNumThreads(1)，落 Harness 构造函数体顶（harness.cpp:444 区），env 门 JMW_OCV_THREADS（默认单线程，"0"＝恢复默认） | 0 diffs（线程数不改位） | 5 对 ABBA |
| B2 | **Z5**(harness move＋dump 门控)→**E1**(缓冲成员化＋boundingRect 手写 cvFloor)→**E4**(bestMask 索引缓存)＋**全内点精确早停**（maxInliers==N break，等变） | 0 diffs | 1 对 |
| B3 | **Z2**(三矩同趟)→**Z4**(Impl 暂存＋cannyEdgeImage 深拷贝＋手扫行主序) | 0 diffs＋M3 dump 逐位 | 1 对 |
| B4 | **Z3**(zernike 点间 AVX2，lane=点保串行链，cpuid 运行时守卫，env 门 JMW_Z3_LEGACY)——**不加 /arch:AVX2**（显式 intrinsic 免门控，避免整 TU 位漂风险） | 0 diffs＋M3 逐位 | 1 对 |
| B5 | **E2**(ellipse 三处内点扫描 SSE2 4 宽：divps 真除保留、两乘一加同序、cmplt、尾部标量同式；阈值常量外提；env 门 JMW_E2_LEGACY) | 0 diffs | 1 对 |
| B6 | **E3**(vendor fitEllipseDirect＋fitEllipseNoDirect＋eigen3：栈缓冲、theRNG() 调用模式逐字保留、×(1.0/5) 写法保留、AᵀA k 升序手写＝DLL 位同；parity gtest ≥10⁴ 组钉 RNG 状态含近奇异；env 门 JMW_E3_LEGACY) | 先逐位；实证不可复刻才降容差（须归档证据＋下游门不放宽） | 5 对 ABBA |
| B7 | 收官：全开关翻转 parity 复跑＋5 对总 A/B＋文档回写 | — | 5 对 |

## 测量协议（终审修正版）

- 逐帧配对（n=399）为主判据：两臂 timing.csv 各步中位/p90＋帧级 Wilcoxon 方向性；轮级均值仅辅助
- ABBA 交替 ≥5 对，每臂丢前 30 帧暖机；轮间冷却 ≥60s；同 Release/同 PATH/同 config_flat＋ALL_raw/同 JMW_* 矩阵（除被测变量）
- A 臂＝归档\工作备份-20261022-zernike-ellipse\speed_replay_base.exe（mtime 对账每轮打印）
- 报告指标：各步中位/p90、wallMk/wall1a/wall1b/wallReg/wallCh2（Z1 全局效应监控）、FPS±SD、golden diffs/flips

## CMake 改动（仅一处，B2 前落地）

三处 GLOB 补 CONFIGURE_DEPENDS（B5/B6 新文件自动入编——否则静默漏编＝终审 P0）；
追加算子自测 glob（ops/*/tests/test_*.cpp 77 用例编入 ctest 门）。

## 回退双轨

- env：JMW_OCV_THREADS / JMW_Z3_LEGACY / JMW_E2_LEGACY / JMW_E3_LEGACY（=1 回旧路径）
- 文件：归档\工作备份-20261022-zernike-ellipse\{zernike_edge,ellipse_fit}\ 整目录还原＋重建
- B7 开关翻转→parity 复跑（没验证过的回退开关等于没有）

## 已剥离/否决项

- CCL bbox 扩边（输出必变，属精度试验另立项）；ROI 拼批 Z6（可选缓）；跨算子 SoA 重构（收益 0.1-0.3ms 缓）；/arch:AVX2 旗标（不需要且添位漂风险）；算子内多线程（lane 绑核封死）；RANSAC 迭代数/早停阈值语义（红线禁区，除等变精确早停外）
