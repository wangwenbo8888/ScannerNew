# framework/ 退役与分层重构 实施方案

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 framework/ 共享层退役，内容按归属迁入 core/ + 11 模块 + app/，建立 `core ← 06 ← 08 ← 07 ← 业务 ← app` 单向分层。

**Architecture:** 见 `docs/plans/2026-08-13-分层与功能划分-design.md`（经六轮代码级核验）。本方案是其落地步骤。每步保持构建绿 + ctest 43/43 绿后提交。

**Tech Stack:** CMake / C++17 / CUDA 17 / MSVC / OpenCV 4.13 / Eigen 3.4.1 / gtest（ctest）。Debug 构建目录 `build/`。

**授权说明：** 本方案涉及 `framework/`（AI 默认受限区）的删除与迁移——经人工多轮明确指令授权（framework优化 全流程）。

**关键约束（来自六轮审查）：**
- 工作流框架（IWorkflow/Pipeline/Stage/RingBuffer）→ **06**（非 app，因 Stage 被模块继承）
- WorkflowContext + 具体工作流 → **app**（WorkflowContext 退役后，stages 改直接接服务指针）
- CalibStore → **01 迁入**（非删，活代码）
- ScannerWindow → **app**（深度耦合 AppContext）
- core/ 只装 types+EventBus+log（铁规）

---

## 预检（执行前一次性）

**Step 0.1：基线绿确认**
```
cmake --build build --config Debug
$env:PATH = 'C:\opencv-cuda-4.13.0-debug\x64\vc17\bin;' + $env:PATH
ctest --test-dir build -C Debug --output-on-failure
```
预期：构建成功 + 43/43 绿。记录基线测试数（迁移全程以此为"未退化"基准）。

**Step 0.2：建 worktree（强烈建议）**
结构性重构，隔离工作树避免污染 master。参考 `using-git-worktrees` 技能。

---

## Phase 1：建 core/ 底座库

**目标：** 新建 `core/`（types.h + EventBus + log），全局 include 改向，framework/common 与 framework/infra 的对应内容**暂时保留**（双轨，下阶段才删）。

### Task 1.1：创建 core/ 目录与 CMakeLists

**Files:**
- Create: `core/CMakeLists.txt`
- Modify: `CMakeLists.txt:108` 前 `add_subdirectory(core)`

`core/CMakeLists.txt`：
```cmake
# core/ — 极薄共享内核（铁规：只装 types + EventBus + log，禁业务代码）
add_library(core STATIC
    EventBus.cpp
)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# EventBus.cpp 暂从 framework/infra 复制（下阶段删 framework 时正式落地）
```

根 `CMakeLists.txt` 在 `add_subdirectory(framework)` 前加 `add_subdirectory(core)`。

### Task 1.2：迁移 types.h 主体到 core/

**Files:**
- Create: `core/types.h`（自 `framework/common/types.h` 抽出主体：Result/QualityFlag/Pose/FaultSeverity/FrameId/TimestampMs）
- 暂留 `framework/common/types.h` 为"转发壳"（`#include "core/types.h"` + 保留 EventType/Event/DeviceState/ScanMode 待后续拆出），保 backward compat

**验证：** `cmake --build build --config Debug` 绿。`framework/common/types.h` 仍被各处 include，转发到 core/types.h，不破坏现状。

### Task 1.3：迁移 EventBus 到 core/

**Files:**
- Move: `framework/infra/EventBus.h/.cpp` → `core/EventBus.h/.cpp`
- `core/EventBus.cpp` 改 `#include "EventBus.h"`（去掉 `infra/` 前缀）
- `framework/infra/EventBus.h` 改为转发壳 `#include "core/EventBus.h"`
- `framework/infra/CMakeLists.txt`：EventBus.cpp 从 fw_infra 源列表移除（fw_infra 暂留 Scheduler/Watchdog 待 Phase 7 删）

**验证：** 构建 + ctest 43/43 绿。
**提交：** `refactor(core): 建 core/ 底座，迁入 types.h 主体与 EventBus`

---

## Phase 2：迁数据层 + 工作流框架 → 06

**目标：** framework/data 容器 + Sink → 06；framework/workflow 的 IWorkflow/Pipeline/Stage/RingBuffer → 06；ParameterManager → 06；CalibStore → 01。

### Task 2.1：迁 framework/data → modules/06

**Files (move):**
- `framework/data/{FrameBuffer,PointCloudBuffer,DeviceStateCache}.{h,cpp}` → `modules/06_fileio/`
- `framework/data/{RingBuffer,IFrameSink,IPointCloudSink,IDeviceStateSink}.h` → `modules/06_fileio/`
- DeviceState 枚举从 types.h 拆出 → `modules/06_fileio/DeviceTypes.h`

**CMake:** `modules/06_fileio/CMakeLists.txt` 由 INTERFACE 改为 STATIC 库（含迁入源 + 现有 file_io.cpp），导出 include。`framework/data/CMakeLists.txt` 清空源（留空壳目录待 Phase 7 删）。app/CMakeLists 的 `fw_data` 链接改 `mod_fileio`。

**验证：** 构建 + ctest 绿。
**提交：** `refactor(06): 迁入 framework/data 容器与 Sink，升格为数据管理库`

### Task 2.2：迁工作流框架 → modules/06

**Files (move):**
- `framework/workflow/IWorkflow.h` → `modules/06_fileio/`（含 WorkflowState/Progress/Callback）
- `framework/workflow/Pipeline.{h,cpp}` → `modules/06_fileio/`（含 Stage 基类）
- RingBuffer 已在 Task 2.1 随 framework/data 迁入 06

**CMake:** Pipeline.cpp 加入 mod_fileio 源。`framework/workflow/CMakeLists.txt` 移除 IWorkflow/Pipeline（仅留具体工作流编译待 Phase 5 处理）。

**include 更新：** 全工程 `#include "workflow/IWorkflow.h"` / `"workflow/Pipeline.h"` → `"IWorkflow.h"` / `"Pipeline.h"`（06 在 include 路径）。涉及：modules/02/ScanWorkflow.h、modules/01/CalibrationWorkflow.h、modules/04/PostProcessWorkflow.h。

**验证：** 构建 + ctest 绿。
**提交：** `refactor(06): 迁入工作流框架(IWorkflow/Pipeline/Stage)，下沉因 Stage 被模块继承`

### Task 2.3：ParameterManager → 06，CalibStore → 01

**Files:**
- Move: `framework/service/ParameterManager.{h,cpp}` → `modules/06_fileio/`
- Move: `framework/data/CalibStore.{h,cpp}` → `modules/01_calibration/`
- 01 的 CMakeLists 由 INTERFACE 改 STATIC（含 CalibStore + 现有 calib_workflow.cpp）；AppContext.cpp 的 `#include "data/CalibStore.h"` 改向 01。

**验证：** 构建 + ctest 绿。
**提交：** `refactor: ParameterManager→06, CalibStore→01（活代码迁入）`

---

## Phase 3：迁设备层 framework/hal → 08

### Task 3.1：HAL 接口 → modules/08

**Files (move):**
- `framework/hal/IScannerCamera.h`、`framework/hal/IMCU.h` → `modules/08_devicemgmt/`
- 08 的 CameraControl.h/MCUDriver.h/HardwareMonitor.h 的 `#include "hal/..."` 改 `"IScannerCamera.h"`/`"IMCU.h"`（08 自身 include 路径）。

**CMake:** mod_devicemgmt 的 include_directories 已含自身目录；确认。framework/hal 清空待删。

**验证：** 构建 + ctest 绿。
**提交：** `refactor(08): 迁入 HAL 接口(IScannerCamera/IMCU)`

---

## Phase 4：迁编排层 framework/service → 07

### Task 4.1：状态机/会话 → modules/07

**Files (move):**
- `framework/service/{StateMachine,SessionService,IState}.{h,cpp}` → `modules/07_session/`
- EventType/Event 从 types.h 拆出 → `modules/07_session/EventTypes.h`
- 07 的 CMakeLists 由占位改 STATIC 库。

**include 更新：** `#include "service/StateMachine.h"` 等 → 07 路径。涉及：modules/02/ScannerWindow.cpp、app/AppContext.cpp。

**CMake:** app 链接加 mod_session。framework/service 暂留 FaultHandler/ParameterManager（ParameterManager 已在 Phase 2 迁走，FaultHandler 待 Phase 6）。

**验证：** 构建 + ctest 绿。
**提交：** `refactor(07): 迁入 StateMachine/SessionService/IState + EventType`

---

## Phase 5：迁装配层（工作流 + ScannerWindow → app，WorkflowContext 退役）

> **本阶段是最大重构**：拆分工作流文件（编排→app，stages→留模块），stages 去 WorkflowContext 化。

### Task 5.1：迁 IWorkflow/Pipeline 框架已在 Phase 2 完成；此处迁具体工作流

**Files (move 编排部分):**
- `modules/02_scanning/ScanWorkflow.{h,cpp}` → `app/`（**拆分**：stages 类定义留 02 新建 `modules/02_scanning/scan_stages.{h,cpp}`，ScanWorkflow 编排类移 app）
- `modules/01_calibration/CalibrationWorkflow.{h,cpp}` → `app/`
- `modules/04_postprocessing/PostProcessWorkflow.{h,cpp}` → `app/`（04 移走后退役为空桩）

### Task 5.2：stages 去 WorkflowContext 化

**Files (modify):**
- `modules/02_scanning/scan_stages.h/.cpp`：每个 stage 构造从 `WorkflowContext* ctx` 改为直接服务指针（`FrameBuffer*`/`EventBus*`/...，由 app 的 ScanWorkflow 注入）
- 同理 01/04 的 stages（若有）

**验证：** 构建 + ctest 绿。⚠ 风险高，stages 行为需回归。

### Task 5.3：ScannerWindow → app

**Files (move):**
- `modules/02_scanning/ScannerWindow.{h,cpp,.ui}` → `app/`
- app/CMakeLists MOD_SCANNING 改为本地源；include 路径调整。

### Task 5.4：WorkflowContext 退役

**Files:**
- 删 `framework/workflow/WorkflowContext.{h,cpp}`
- `app/AppContext.cpp`：去掉 wfCtx_ 装配，服务直接注入各工作流构造函数。
- app/CMakeLists 移除 fw_workflow 链接（其内容 IWorkflow/Pipeline 已在 06，工作流在 app 本地）。

**验证：** 构建 + ctest 绿。
**提交：** `refactor(app): 工作流+ScannerWindow 归 app，WorkflowContext 退役，stages 去_ctx 化`

---

## Phase 6：迁故障处理 FaultHandler → 10

### Task 6.1

**Files (move):**
- `framework/service/FaultHandler.{h,cpp}` → `modules/10_observability/`
- 10 的 CMakeLists 由占位改 STATIC。
- AppContext.cpp 的 `#include "service/FaultHandler.h"` 改向 10。

**验证：** 构建 + ctest 绿。
**提交：** `refactor(10): 迁入 FaultHandler`

---

## Phase 7：删 framework/ 整树 + 死代码

### Task 7.1：删死代码（先于整树，确保无引用）

**Files (delete):**
- `framework/infra/{Scheduler,Watchdog}.{h,cpp}`
- `framework/algorithm/operator_convention.h`
- `framework/crosscut/IAuth.h`
- `framework/ui/IView.h`（及 framework/ui 目录）
- `framework/common/common.cpp`（空桩）
- `framework/{tests,workflow/tests,data/tests}/test_*.cpp`

**CMake:** framework/infra/CMakeLists 移除 Scheduler/Watchdog；framework/{algorithm,crosscut,ui,tests}/CMakeLists 清空。

**验证：** 构建 + ctest 绿（确认这些真无生产引用——六轮已 grep 核实）。

### Task 7.2：删 framework/ 整树

**Files:** 删 `framework/` 整个目录（含转发壳 types.h/EventBus.h 等，此时全工程已改向 core/06/07/08）。
**CMake:** 根 CMakeLists 移除 `add_subdirectory(framework)`。

**验证：** 构建 + ctest 43/43 绿（与基线同）。
**提交：** `chore: 删除 framework/ 整树（退役完成）+ 死代码`

### Task 7.3：更新 AGENTS.md / 工程目录地图

`framework/AGENTS.md` 删除；`AGENTS.md`/`工程目录地图.md` 移除 framework/ 段，补 core/ 层；`开发进度.md` 记录退役完成。

**提交：** `docs: framework/ 退役后同步架构文档`

---

## Phase 8（独立工作项，非本方案范围）

下列已有独立设计/实施方案，framework 退役后另行排期：
- **设备顶层 FSM 7 态演进** → `docs/plans/2026-08-10-设备状态机-实施方案.md`（落 07）
- **按键交互 FSM + MCUDriver 协议升级** → `docs/plans/2026-08-12-按键交互协议-v2-design.md`（落 08）
- **外设链路 FSM** → 待设计（落 08）
- **WorkflowArtifactStore（L4 仓库）** → `docs/模块功能/06-文件管理.md §3.3`（落 06）

---

## 风险与回滚

| 风险 | 缓解 |
|---|---|
| Phase 5 工作流拆分 + stages 改造行为回归 | 拆分后跑全量回归；保留拆分前 commit 便于对比 |
| framework/ 转发壳期间双轨混乱 | Phase 1-6 保留转发壳保 backward compat；Phase 7 才删 framework/ |
| include 路径遗漏致编译错 | 每阶段构建验证；用 `cmake --build` 的错误信息定位遗漏 |
| ctest 测试数下降 | 每阶段 ctest，低于基线 43 即停查 |

**回滚单位：** 每个 Phase 一个或多个 commit，可 `git revert` 单 Phase。

---

## 待决（执行前需确认）

1. **是否用 worktree 隔离**（强烈建议——大重构）
2. **Phase 5 stages 拆分粒度**：ScanWorkflow.cpp 当前 stages 与编排同文件，拆分边界以 stages 类为单元还是更细
3. **core/ 命名**：六审提出与 09/core/ 撞名，建议改名（base/foundation）——是否在 Phase 1 落地改名，还是保留 core/ 称谓
