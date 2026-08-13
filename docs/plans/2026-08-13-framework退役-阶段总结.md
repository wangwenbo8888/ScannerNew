# framework/ 退役与分层重构 — 阶段总结

> **日期**: 2026-08-13
> **分支**: `feature/base-foundation`（8 提交，**全程构建绿 + ctest 43/43**）
> **状态**: Phase 1/2/3/4/6 完成；Phase 5/7 待续（依赖 Qt5 Svg/SerialPort 补全）
> **定位**: 本次"按键交互协议 → 设备管理状态机 → framework 退役"整条工作线的归档总结。

---

## 一、背景

原架构三层 `framework/ + modules/ + app/` 中 `framework/` 沦为"无主共享杂物间"。本工作线目标：退役 framework/，按归属迁入 `core(base)/ + 11 模块 + app/`，建立 `base ← 06 ← 08 ← 07 ← 业务 ← app` 单向分层。

---

## 二、产出文档清单

### 设计文档（`docs/plans/`）
| 文档 | 内容 | 状态 |
|---|---|---|
| `2026-08-12-按键交互协议-design.md` | 按键协议 v1.0（已被 v2 取代，保留历史） | 历史 |
| `2026-08-12-按键交互协议-v2-design.md` | **按键协议 v2.0**：定长 `K<ID><S|D|H>;`，MCU 判手势/PC 做语义，12 报文映射 | 定稿 |
| `2026-08-13-分层与功能划分-design.md` | **分层总方案**（6 层 + 总表 + 文件级明细 + 4 状态机落点） | 定稿（六轮审查） |
| `2026-08-13-framework退役-实施方案.md` | **8 阶段迁移实施计划**（Phase 0-8 + TDD 步骤） | 执行中 |

### 权威落地（模块文档）
| 文档 | 内容 |
|---|---|
| `docs/模块功能/08-设备管理.md §3.1` | 按键交互协议 v2.0 权威落地（7 子节 + 通俗附录） |

### 既有相关（未改）
- `docs/plans/2026-08-09-设备状态机-design.md` + `2026-08-10-设备状态机-实施方案.md`（顶层 FSM 7 态，落 07）
- `framework优化.md`（原始提议，本方案是其修正版）

---

## 三、六轮代码级审查的关键纠正

每轮靠 grep/read 实证（非记忆/推断），累计纠正：

| 轮 | 纠正 |
|---|---|
| 2 | CalibStore 非孤儿（AppContext 创建+注入）→ 删改**迁**；ScanConfig.h 非已有→**新建** |
| 3 | 工作流框架→app 制造循环依赖（修法 B）；04_postprocessing 非空（glob 拼错目录名） |
| 4 | ScannerWindow 深度耦合 AppContext → 上移 app（选项 A）；01 UI 经核实不耦合 |
| 5 | 04 无独立 stages；stages 耦合 WorkflowContext（修复 B 代价被低估）；core/ 与 09/core/ 撞名 → 改名 **base/** |
| 6 | Stage 基类在 Pipeline.h 被模块 stages **继承** → 框架须下沉 06（非 app） |

**核心教训**：每个"X 归 Y"断言必须核验 X 的**所有依赖方**（谁继承/持有 X）。

---

## 四、分支成果（8 提交，每步构建绿 + ctest）

```
7cf65d2  Phase 6：FaultHandler → mod_observability(10)
9f5c821  Phase 4：service(StateMachine/Session/IState) → mod_session(07)
0669bd8  Phase 3：hal(IScannerCamera/IMCU) → mod_devicemgmt(08)
9fe31ae  Phase 2c：ParameterManager→06, CalibStore→01
ab233e4  Phase 2b：IWorkflow/Pipeline(Stage) → mod_fileio(06)
23130c8  Phase 2.1：framework/data 容器+Sink → mod_fileio(06)
88eb5f5  Phase 1：base/ 底座（types.h + EventBus）
3fc8617  基线：核心实现 + CMake 集成修复
```

### 基线修复（3fc8617 之前的工作树缺陷）
工作树是"未完成集成的大重构"，CMake 没接好线：
- 根 CMake：去掉已删的 `add_subdirectory(sdk)`
- app/CMake：`find_package(Qt5)` + 前缀路径 + 修 rcc 路径；scan_demo **条件构建**（缺 Qt5 Svg/SerialPort 时跳过）
- modules/08 CMake：Galaxy SDK 路径指向真实安装

---

## 五、当前架构状态

### 新分层落地情况
| 层 | 目录 | 装入内容 | 来源 |
|---|---|---|---|
| ① 底座 | `base/`（新建） | types.h + EventBus | framework/common + framework/infra |
| ② 数据 | `modules/06_fileio`（升 STATIC） | FrameBuffer/PointCloudBuffer/DeviceStateCache + 三 Sink + RingBuffer + IWorkflow/Pipeline(Stage) + ParameterManager | framework/data + framework/workflow + framework/service |
| ③ 设备 | `modules/08_devicemgmt` | + IScannerCamera/IMCU 接口 | framework/hal |
| ④ 编排 | `modules/07_session`（升 STATIC） | StateMachine/SessionService/IState | framework/service |
| ⑤ 业务 | 各模块 | CalibStore→01 | framework/data |
| ⑥ 可观测 | `modules/10_observability`（升 STATIC） | FaultHandler | framework/service |

### framework/ 残余（Phase 7 删）
| 子目录 | 内容 |
|---|---|
| common/ | types.h **转发壳** → base/types.h |
| infra/ | EventBus.h **转发壳** → base/EventBus.h + Scheduler/Watchdog（**死代码**） |
| data/ hal/ service/ | 空 INTERFACE 壳 |
| ui/ algorithm/ crosscut/ | 死代码（IView/operator_convention/IAuth） |
| **workflow/** | **WorkflowContext + 3 模块工作流**（fw_workflow 编译）— Phase 5 目标 |

### 全局 include 路径（根 CMake）
`${CMAKE_SOURCE_DIR}` + `modules/{06_fileio, 07_session, 08_devicemgmt, 10_observability}`

---

## 六、验证状态

- **库 + ctest：43/43 全绿**（删 4 个 dead framework 测试后；原 46）
- **scan_demo（Qt app）：跳过构建**——缺 Qt5 Svg/SerialPort。这是 Phase 5 半不可验证的根因。

---

## 七、剩余工作（Phase 5 + 7）

### Phase 5（最大，且半不可验证）
- 工作流框架已在 06（Phase 2b 完成）
- 待做：ScanWorkflow/CalibrationWorkflow/PostProcessWorkflow（fw_workflow，**可验证**）→ app 并拆分；stages 去 WorkflowContext 化
- ScannerWindow→app、AppContext 改 DI（scan_demo，**不可验证**直到 Qt 装好）
- WorkflowContext 退役

### Phase 7（依赖 Phase 5）
- 删 framework/ 整树 + 死代码 + 转发壳
- 同步 `AGENTS.md`/`工程目录地图.md`/`开发进度.md`

### 续作前置
**装 Qt5 Svg/SerialPort**（用 Qt 维护工具，或装 Python 后用 aqtinstall），使 scan_demo 可构建，Phase 5 才能全量验证。

---

## 八、关键教训（已记入提交）

1. **PowerShell `Set-Content -Encoding UTF8` 腐蚀 `#`→`i`**（Phase 4 踩过）——后续内容改写一律用 **Edit/Write 工具** 或 `git checkout HEAD` + 二进制 `Move-Item`。
2. **执行前必须核验基线**：本以为"43/43 绿仓库"，实际是未提交半成品 + CMake 未接好线 + 缺 Qt 模块。多轮 grep/read 才摸清。
3. **转发壳双轨**：Phase 1-6 保留 framework/ 转发壳维持 include 兼容，Phase 7 才删——避免迁移中途大面积破坏。

---

## 九、续作命令

```powershell
git checkout feature/base-foundation
cmake -S . -B build; cmake --build build --config Debug
$env:PATH = 'C:\opencv-cuda-4.13.0-debug\x64\vc17\bin;' + $env:PATH
ctest --test-dir build -C Debug   # 确认 43/43 基线
# 装 Qt5 Svg/SerialPort 后，从 Phase 5 续作（见 framework退役-实施方案.md §Phase 5）
```
