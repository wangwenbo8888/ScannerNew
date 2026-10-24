# 产线集成 speed_optimize · 工作日志（Phase 0）

> 逐项记录每笔修改，保证可追溯/可回滚。计划与决策点见同目录《2026-10-24-产线集成speed_optimize-计划.md》，文件级差异见《2026-10-24-产线集成speed_optimize-差异台账.md》。

## 2026-10-24 · Phase 0 完成（回放验收基建＋源码纳管＋台账冻结）

### 修改 1：speed_optimize 源码入库（286 文件，87,835 行）
- **内容**：`modules/09_operatorlib/TEST/speed_optimize/` 全量首次提交——`src/`（五链执行器 fc_executor＋harness 等 25 文件）、`ops/`（算子最终形态拷贝）、`tests/`、`cmake/`、`config*.json`、`README.md`、`接力.md`、`归档/记录文档/`（43 份实验记录）、`文件校验清单.csv`
- **数据外置**：`.gitignore` 增补 `归档/实测数据/`、`compare/`、`date_in/`、`golden/`、`out*/`、`*.ply`、`*.exe`——4.8GB 帧数据＋150MB golden＋288MB 对拍图不入库，磁盘原位供回放
- **动机**：最终形态源码此前零版本管理（接力.md 明示风险），是后续一切合入的唯一基准

### 修改 2：差异台账生成（docs/plans/…-差异台账.md）
- **方法**：MD5 逐文件对照 `ops/` ↔ `modules/OperatorLib/`（非 tests、非 global_optim）
- **结论**：**同 34｜异 49｜ops 独有 45｜主库独有 52**（计 180 文件）
- 处置代号：A=三方合并（保主库修复）/ B=ops 新增直接入 / C=主库保留 / D=随五链 Phase 2 定
- Phase 1 分 5 批依据已入台账

### 修改 3：构建接线（根 CMakeLists.txt）
- 新增 `option(BUILD_SCAN_REPLAY "Build speed_optimize replay harness (golden 对拍验收, Release-only)" OFF)`，ON 时 `add_subdirectory(modules/09_operatorlib/TEST/speed_optimize)`
- **默认 OFF＝主库构建零影响**（已验证）

### 修改 4：沙盒 CMake 三方库守卫（speed_optimize/CMakeLists.txt）
- spdlog/nlohmann_json/googletest 的 FetchContent 全部加 `if(NOT TARGET ...)` 守卫——联建时主库同名 target（spdlog shared/json/gtest）已存在，防 target 重定义
- 其余不动（nvcc pin/CUDA arch 与主库同值；project() 子目录嵌套合法）

### 修改 5：根 .gitignore
- 增 `build-so/`（联建专用目录）

## 验证记录（全绿）

| 项 | 结果 |
|---|---|
| 主库默认构建（OFF）Release | ✅ 全目标编译通过 |
| 主库 ctest | ✅ **100/100** |
| 联建配置（build-so，ON） | ✅ configure 68s 通过（FetchContent 守卫生效） |
| speed_ops 算子库（~90 源文件含 .cu） | ✅ 编译通过 |
| speed_replay / test_speed_replay | ✅ 两个 exe 产出（17/18MB） |
| speed_replay --help | ✅ 正常输出（16 算子 v3 链） |
| test_speed_replay | ✅ **72 用例全过** |

## 联建命令（后续验收标准口令）

```powershell
# 配置＋构建（独立目录，不动主 build/）
cmake -S . -B build-so -G "Visual Studio 17 2022" -A x64 "-DCMAKE_CONFIGURATION_TYPES=Release" "-DBUILD_SCAN_REPLAY=ON" "-DJMW_GH_MIRROR=https://ghfast.top/"
cmake --build build-so --config Release --target speed_replay test_speed_replay
# 冒烟
$env:PATH = "F:\opencv4.13\install\x64\vc17\bin;build-so\bin\Release;$env:PATH"
build-so\bin\Release\test_speed_replay.exe            # 72 用例
# 399 帧 golden 对拍（Phase 1 每批验收红线）
build-so\bin\Release\speed_replay.exe --config <config_flat.json> --dump-dir <out> …
```

## 下一步（Phase 1 · 算子合入，5 批）

按台账分批：①core/laser 新件＋SIMD ②v3 匹配族＋curve_map ③fuse/mask/zernike/ellipse 三方合并 ④其余异文件 ⑤调度类型。每批验收＝单测＋399 帧 0 diffs＋主库 ctest 100/100。

---

## 2026-10-24 · Phase 1 批 1 完成（core/laser 新件＋marker 类型增量＋ODR 修复）

### 换入清单（ops → modules/OperatorLib）
**新增算子（B）**：`core/laser/steger_fast`（4 文件＋test）、`core/laser/flat_center`（4＋test 同名 flat 相关在 72 用例内）、`core/laser/epipolar_interp_dual`（4）、`core/marker/conic_reconstruct`（2＋test）、`core/marker/ellipse_fit/ellipse_fit_{simd,vendor}.{h,cpp}`、`core/laser/laser_reconstruct/laser_reconstruct_cpu.{h,cpp}`、`core/common/{steger_types.h,epipolar_types.h}`、`core/common/testutil/marker_synth.{h,cpp}`
**整文件替换（A-增量）**：`core/common/zitai_result_types.h`（+EllipseFitCPUResult::groupId、+MarkerReconstructResult::message）、`core/common/calib_result_types.h`（注释更新）、`core/marker/point_reconstruct/point_reconstruct_cpu.{h,cpp}`（+整组质量闸 4 参数）、`core/marker/ellipse_fit/ellipse_fit_cpu.{h,cpp}`（+内点率复检/形状闸/输出过滤）；两组件 tests 同步换 ops 版

### ODR 修复（本批最大收获）
- **现象**：steger_fast 四个 Smoke 测试抛 `StegerParams::maxLabels must be [1,4096]`（maxLabels=0 幽灵值）
- **根因**：`StegerParams/GroupMode/StegerResult` 三类型在旧 `steger/steger_extract_cuda.h` 与新 `common/steger_types.h` **双定义**——链接器 COMDAT 择一，新算子内联链拿到旧 validate＋旧布局（字段错位读出 0）
- **修法**：旧 steger 头退役本地三定义改 `#include "common/steger_types.h"`（沙盒最终形态本就如此——ops 已删除旧 steger 算子；主库保留旧算子（07 仍在用，D3 一个周期），但类型单一驻扎）
- **连带排查记录**：中途 4 起测试失败为**排查插桩自身 bug**（`if(...) fprintf(...); throw...;` 未加花括号→throw 无条件执行）——教训入册：插桩多语句必须花括号；以及 `CMAKE_CUDA_DEPENDENCIES_USE_COMPILER=OFF`（根 CMake 继承沙盒口径）→ **.cu 不跟踪头变更，改头后须 touch 相关 .cu 强制重编**

### 验证（双绿）
- 主库 Release＋Debug ctest **102/102**（原 100＋新增 test_steger_fast/test_conic_reconstruct_cpu）
- 混链陈旧二进制坑（DLL 边界头布局变更后部分目标未重编→SEH 崩溃假象）：**头结构变更后必须全量重建**，已记入易踩坑

### 批 1 消化的台账项（原批 3/4 前移）
zitai/calib 两类型头、point_reconstruct、ellipse_fit_cpu——conic 的依赖链要求，按"沙盒为源"整组换入

### 下一步
批 2：laser_match_scan_v3 三变体＋cmtt_table_loader＋curve_map 系 8 文件（依赖 cmtt 容器，预计牵出 calibration 域小改）

---

## 2026-10-24 · Phase 1 批 2＋批 3 完成（v3 匹配族＋curve_map 系＋三方合并五算子）

### 批 2（零摩擦）
**新增（B）**：`scanning/laser/{laser_match_scan_v3(8文件，含 cmtt_table_loader＋split_front.cu), laser_match_scan_v3_cpu(2), laser_match_scan_v3_split(3)}`＋`calibration/laser_calib/{curve_map(3), curve_map_temp_table(11，含 cmtt_container＋temp_param_interpolator)}`；7 新测试随入（acceptance/golden_ref_tier/cmtt_container/curve_map_temp_table/temp_param_interpolator/cmtt_table_loader/curve_map）。**一把过**——批 1 公共类型垫层生效。109/109 双绿。

### 批 3（三方合并重头戏：17 文件换入＋1 保留＋4 处适配）
**换入 ops 版**：mask_separation×4（P0 八核形态学＋CUDA Graph 首次进主库！）、laser_cloud_fuse_cuda×4（取模探测＋rehash 扩容＋survivor 统计）、zernike_edge_cpu.cpp（P2 SIMD +207 行）、ccl×4（topX 面积收集 GPU 化——退役 12.6MB pageable D2H＋CPU 315万标签扫描）、mask_extract×4（erode≤1 恒等旁路＋滤波器预建）
**保留主库版**：laser_reconstruct_cuda（主库 261018 pinned 回读＋流序 copyTo 优于 ops 近基线版）
**手工合并**：ccl 回补 261018 legacy 栅栏（ops 版无——OpenCV CC 无 stream 重载，非阻塞调用流保序必需；ev_legacy_done_ 事件＋pimpl 成员）
**四处适配**（ops 接口变更的下游对齐）：
1. laser_cloud_normal_cuda（主库独有算子）：DeviceContext `mask`→`capacity` 退役——lookupVoxelDevice 探测方案整体重写为 `h % capacity` 线性探测＋hash64(key)（原 `h & mask` 幂掩码——与 ops fuse 插入路径不匹配＝全 miss→processedCount=0 三连失败的根因）
2. ScanChains.cpp：`spotThreshold`→`threshold` 更名；`d_gray` 直通口退役回退直传（每帧多 6MB PCIe，Phase 2 消）
3. fuse_cuda 测试：`ctx.mask==1023` 断言→`capacity>=1`（占位防呆）
4. mask_sep 测试：WarmupAndTimings 断言条件化——CUDA Graph 路径 timings 恒 0 是预期（P0 优化本质），仅 JMW_SEPTIME=1/legacy 回退才计时
**测试换 ops 版**：zernike/mask_extract×2/mask_sep

### 验证
Release＋Debug ctest **109/109 双绿**＋LeadScannerK2 全量编译通过

### 批 3 消化的台账项
原批 3 全部（fuse/mask/zernike/ellipse——ellipse 已在批 1）＋原批 4 的 ccl/mask_extract/undistort（undistort 换入在批 3 的 17 文件内）

---

## 2026-10-24 · Phase 1 批 4 完成（剩余异文件清账＋标定链温度表口径适配）

### 换入 ops 版（25 文件＋6 测试）
stereo_rectify_temp_table×2、laser_label×3、laser_match(calib)×2、projector_joint_calib×2、temp 三补偿×6、calib_result_types、undistort_cuda、edge_match×2、frame_fuse×2、marker_match、optical_flow_fuse×2、prev_frame_state.h；测试换 ops：laser_label/edge_match/frame_fuse/marker_match/optical_flow_fuse/stereo_rectify_temp_table/intrinsic_compensate 等随源

### 下游适配（四处）
1. **ScanChains**：`useNormalWeight=true` 显式置位（ops 新校验 normalAngleThresh≥30 须 useNormalWeight——否则 lane 算子集构造即抛）
2. **温度表口径**（ops 默认 0.5/-15..15 → 01 标定链显式保持 0.2/-10..10=101 条）：CameraChain 5-1 Intrinsic/5-2 Extrinsic/3-5 StereoRectifyTempTable＋LaserChain 5-3 LaserExtrinsic 四处显式传参（查表/测试口径依赖 101 条；ops 新默认保留给运行时扫描端）
3. **frame_fuse 测试**：addNoise sigma=0 → 钳 1e-12（MSVC Debug normal_distribution 断言 sigma>0，沙盒只跑 Release 未踩）
4. **调试插曲**：laser_label "expectedLineCount must be [0,maxLabels]" 幽灵异常＝**陈旧 .obj**（tlog 依赖失灵）——`build/modules` 全 obj 清除重编即愈；此坑已两次踩（批 3 mask_sep 同病），规程化：**头文件改动后必须清 obj 或 touch 全部相关源**

### 验证
Release＋Debug ctest **109/109 双绿**＋LeadScannerK2 编译通过

### Phase 1 收官状态
ops→主库差异清零（除刻意保留项：laser_reconstruct×2 主库 261018 版、plane_map 系/旧 steger/laser_match v1/fusion CPU+normal 件 D3 保留、example_usage 不入构建）。**台账 49 异文件全部处置完毕**。



