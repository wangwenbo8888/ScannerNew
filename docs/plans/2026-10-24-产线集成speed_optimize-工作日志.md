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
