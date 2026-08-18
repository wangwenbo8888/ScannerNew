# DeviceManager 设备管理器 · 设计文档

> 2026-08-18 brainstorming 产物（人工逐节确认：①结构门面 ②内部逻辑件 ③生命周期线程 ④故障异常 ⑤测试验收 全 OK）。
> 基准：`docs/模块功能/01-标定工作流.md` / `02-扫描工作流.md` 🔒 定稿、`模块功能目录.md §三` 设计对象功能基准、`docs/模块功能/08-设备管理.md`（含 2026-08-18 人工五条修正与上行架构定案）。
> 状态：**设计定稿，未实现**。落地时以本文档为准。

---

## 0. 背景与裁决记录

DeviceManager 是 08 的聚合根、【公】app 存活件（开机即造、全程不销毁），08 对外唯一门面。本文档由以下已确认裁决驱动：

| 裁决 | 内容 |
|---|---|
| 人工修正① | CameraControl 只设相机参数；触发开闭由下位机硬触发（MCUDriver 下发） |
| 人工修正③ | KeyManager 只判「哪个键+短/双/长击」；全部逻辑判断归 DeviceManager |
| 人工修正④ | 3 号下位机灯裁撤；下位机状态由 DeviceManager 记录 |
| 人工修正⑤ | DeviceManager 总负责：收集信息→判断→发送执行命令 |
| 上行架构定案 | 传输层合并（一条串口一个分帧器）；PC 入口按帧头分流；处理不合并；汇合点在 DeviceManager；温度改 MCU 主动上报 |
| 本设计 6 问裁决 | 内部逻辑件拆分（非单类）；自持模式变量+门禁联动；内置预热状态机；故障只报不处；自持逻辑线程；平级小组件文件 |

---

## 1. 结构总览与对外门面

### 1.1 文件组织（`modules/08_devicemgmt/`，全部编入 `mod_devicemgmt`）

```
DeviceManager.h/.cpp      ← 门面本体（对外唯一窗口，~200 行以内）
MenuLogic.h/.cpp          ← 菜单状态机（layer/cursor/adjustCtx + parity 计数）
KeySemantics.h/.cpp       ← 按键语义映射（手势+菜单态+门控 → KeyAction）
WarmupSequence.h/.cpp     ← 预热状态机（加热→采温→热平衡判定→超时保护）
ModeController.h/.cpp     ← 模式变量（idle/calib/scan）+ 门禁联动查询
（已有）CameraControl / MCUDriver / KeyManager / HardwareMonitor
```

### 1.2 对外门面接口（按 01/02 步骤号标注）

| 方法 | 作用 | 服务于 |
|---|---|---|
| `open()/close()` | 一条龙开关机：枚举相机→开相机→开串口→起监控→起逻辑线程 | app 启动/退出 |
| `enterCalibration()` / `enterScan()` / `toIdle()` | 切模式（内部：查全局门禁→ModeController 切变量→MCU 相应命令） | 01-②；02-②；⑨/⑩ |
| `startWarmup(targetC, onReady)` | 预热至热平衡，完成回调 | 01-③ |
| `getDeviceState()` | 设备就绪状态汇总（相机+串口+模式）——02-① 门禁「设备检查」、01 §4 自检的查询口 | 01-①；02-① |
| `getLastTemperatures()` | 最近 4 路+相机温度快照（含时间戳，消费方判新鲜度） | 01-④；02-④ 帧温来源 |
| 事件（EventBus 广播） | `TemperatureUpdate` / `KeyActionFired` / `DeviceModeChanged` / `DeviceFault` | 全工程订阅 |

**关键点**：对外不暴露任何子对象指针（严门面）；门禁查询经回调注入（`GateQuery`，指向 app 状态机——不反向依赖 07/10）。

---

## 2. 内部逻辑件（四个小组件）

### 2.1 ModeController 模式管家

- 状态：`enum class DeviceMode { Idle, Calibrating, Scanning }`——三值，原子读写
- 切换规则：切换前必查 `GateQuery`：门禁不放行→拒绝并返回原因；放行→切变量→DeviceManager 发对应 MCU 命令→广播 `DeviceModeChanged`
- 「记录下位机状态」落点：本组件即原 3 号灯职责的载体——模式是 PC 侧对设备状态的**记账**（下位机本身无档位），任何代码问「设备现在在干嘛」都查它

### 2.2 WarmupSequence 预热状态机

```
Idle → Heating(发加热命令) → Soaking(周期采温)
     → 判定: 温度变化率 ≤ 0.1℃/连续10s 且 距目标 ≤ 2℃ → Balanced(回调 onReady)
     → 超时(默认15min) → Timeout(回调 onTimeout, 不自作主张停加热——报人工/01)
```

- 温度来源：订阅温度链（MCU 主动上报），不自己轮询
- 参数（目标温度/变化率阈值/超时）构造时注入可调——不同机型可配
- 只服务标定预热；异常温度（超安全上限）不在它判——归故障链（§4）

### 2.3 MenuLogic 菜单状态机（原 4 号灯主体）

- 状态：`layer(1|2)` + `cursor(①..④)` + `adjustCtx(none|视野|亮度)` + `parity[M][S]/[U][S]/[U][D]` + `modeCursor(1|2/3)`
- 输入：KeySemantics 给的「语义动作」（ToggleMenu/MoveCursor/Select/Adjust±…），输出：菜单状态变化 + `KeyMenuChanged/KeyModeChanged` 广播（app UI 渲染）
- **纯状态机**：不判门控、不发 MCU 命令——输入进来的都合法（门控在上游）

### 2.4 KeySemantics 按键语义映射（总判断的按键部分）

```
输入: KeyManager 的手势事件(键+短/双/长击)
  ↓ ①查门控: 键类×当前全局态(S2/S4/S5 放行集, M1 分类门控)
  ↓ ②查菜单态: MenuLogic 当前 layer/cursor/adjustCtx
  ↓ ③查 parity: 定 toggle 方向
输出(三选一):
  · MCU 直令: 采集启停→N11 H1/H0(02-D2, 08 自主)
  · MenuLogic 变更: 菜单/模式切换(本地)
  · 工作流命令: 「扫描完成③」「开始后处理④」→ 经状态机门禁→派工作流
丢弃: 门控不放行→丢弃+计数不增+记日志
```

- 三个查询源（门禁/菜单/parity）+ 三种出口（直令/本地/派工作流）——「收集→判断→发令」的完整落地
- 非生效态丢弃规则沿用 M1 定案（采集启停键 S2+S4/S5；菜单/模式/调节键仅 S2）

**依赖方向**：KeySemantics→MenuLogic（单向查）；两者都不碰 MCU/相机——发令统一经 DeviceManager 编排。

---

## 3. 生命周期与线程

### 3.1 `open()` 启动序列（任一步失败→逆序回滚并报 `DeviceFault`）

```
1. 枚举相机(找不到→失败)
2. CameraControl::open(SN 独占) ──失败→ 回滚(1)
3. MCUDriver::open(串口)        ──失败→ 回滚(2→1)
4. KeyManager 启动(挂到 MCUDriver 按键分流)
5. HardwareMonitor 启动(在线/帧率巡检)
6. 逻辑线程启动(见 3.2)
7. 广播 DeviceReady
```

- 全程**只由 AppContext 调一次**（app 启动时）；启动即处于 `Idle` 模式；相机参数按配置文件下发

### 3.2 线程全景

| 线程 | 谁的 | 干什么 | 禁止 |
|---|---|---|---|
| MCU 接收线程（已有） | MCUDriver | 分帧→按帧头分流：温度直接广播；**按键只入队** | 不做任何业务逻辑 |
| 逻辑线程（新增，唯一） | DeviceManager | 串行处理：按键队列消费→KeySemantics→MenuLogic→发令/派发；预热状态机 tick | 不阻塞等待（命令异步发） |
| 监控线程（已有） | HardwareMonitor | 1000ms 巡检在线/帧率 | 不做阈值判断 |
| 调用方线程 | app/工作流 | 调门面方法（切模式/预热/查询） | — |

**设计要点**：所有按键逻辑串行在**一个**逻辑线程——天然免锁（菜单/parity 无并发写）；MCU 接收线程永不阻塞（按键队列有界，满则丢最旧+warn，靠 KeyManager 时序对账发现）。

### 3.3 模式切换与预热的线程关系

- `enterCalibration()`：调用方线程发起点→逻辑线程执行（查门禁→ModeController 切→MCU 命令）→完成经回调/事件回话
- 预热：逻辑线程温度事件驱动 tick WarmupSequence（非 sleep 轮询）；`onReady` 回调在工作流等待点触发
- `close()` 逆序：停逻辑线程→停监控→关 KeyManager→关串口→关相机；进行中的预热/采集直接终止并记日志（app 退出场景，无需优雅）

### 3.4 AppContext 装配收拢（迁移项）

现状 `AppContext.cpp:52-83` 自 new 三零件并布线——落地后收拢进 `DeviceManager::open()`，AppContext 只剩：

```cpp
deviceManager_ = std::make_unique<DeviceManager>(cfg, gateQuery, eventBus);
deviceManager_->open();
```

现有 ScannerWindow/MainWindow 直接拿 CameraControl 裸指针的 4 处（`ScannerWindow.cpp:42,62` / `MainWindow.cpp:200,251,439`），改为经 DeviceManager 门面方法访问。

---

## 4. 故障与异常处置

**总原则（只报不处）**：DeviceManager 检测异常→打包 `DeviceFault`→上报 10 `reportFault`，**不执行任何安全动作**；处置决策由 10 FaultHandler 做出后经普通门面命令回来执行（如 `toIdle()`）。唯一例外：`close()` 清理。

### 4.1 异常一览

| # | 异常 | 谁检测 | 怎么发现 | 上报内容 |
|---|---|---|---|---|
| 1 | 相机掉线 | HardwareMonitor | 巡检 `isOpen()` 变 false / 帧率归零 | 设备ID+断连时刻 |
| 2 | 串口超时 | MCUDriver | 收帧间隔超阈值（如 10s 无上报） | 最后收帧时刻 |
| 3 | 温度超限 | 温度链 | 上报值超安全上限（构造注入阈值） | 哪一路+当前值 |
| 4 | 温度异常变化 | 温度链 | 变化率超限（如 >2℃/s） | 哪一路+速率 |
| 5 | 预热超时 | WarmupSequence | 状态机 Timeout 态 | 目标温度+实际末温 |
| 6 | 按键队列溢出 | 逻辑线程 | 有界队满丢旧 | 丢弃计数 |
| 7 | MCU 命令无响应 | DeviceManager | 发令后等 ACK/状态回报超时（协议支持时） | 命令号+等待时长 |

### 4.2 降级行为（上报之外允许做的）

- **重试**：open 序列/命令下发可重试 1 次；再失败才升级为异常上报
- **缓存续报**：温度链停更时 `getLastTemperatures()` 返回最后有效值+时间戳（不造数据、不清零）
- **门面查询诚实**：`getDeviceState()` 如实反映离线/未知

### 4.3 不归它管的（防越界）

安全停机策略→10；用户提示 UI→app（订阅 `DeviceFault`）；故障日志持久化→10。

### 4.4 测试钩子

内部逻辑件全部可注入 Mock：`GateQuery` / MCU 收发（假串口回灌帧）/ 温度源（合成温度序列）；异常注入点：断连/超时/超温各一个开关——供 10 故障链联调与自动化回归。

---

## 5. 测试与验收

### 5.1 策略——纯 PC 逻辑全自动化，真机只做冒烟

| 层 | 测什么 | 怎么测 |
|---|---|---|
| 单元测试（gtest，入 ctest） | 四个逻辑件各自状态机 | Mock 注入，无真机 |
| 集成测试 | 门面编排：open 序列/切模式/预热/按键全链 | MockCamera + 假 MCU（回灌帧脚本） |
| 联调冒烟（人工，真机） | 串口协议对齐/真实预热时长/按键手感阈值 | 手动清单，不做 CI |

### 5.2 核心用例（8 条）

| # | 用例 | 验证点 |
|---|---|---|
| T1 | open 序列第 3 步串口失败 | 逆序回滚（相机已关）、DeviceFault 上报、无泄漏 |
| T2 | enterScan 在门禁拒绝时 | 模式不变、返回原因、无 MCU 命令发出 |
| T3 | 预热：合成温度先升后稳 | 变化率达标→onReady 恰好触发一次 |
| T4 | 预热：温度永不稳 | 超时→onTimeout（不停加热、不崩溃） |
| T5 | 扫描中按菜单键（非 S2） | 丢弃+计数不增；KMS 采集键 S4/S5 放行→N11 正确翻转 |
| T6 | 菜单两层遍历：4 键 × 3 手势 × layer1/2 | parity/cursor/adjustCtx 全状态机覆盖（表驱动） |
| T7 | 按键队列打满（burst 100 条） | 丢最旧+warn+对账计数正确，逻辑线程不阻塞 |
| T8 | 采集中断连（Mock 相机中途 isOpen=false） | DeviceFault 内容正确；**无**任何自主停采动作 |

### 5.3 验收标准

- T1–T8 全绿（ctest 可重复跑）；逻辑线程无 data race（TSAN 或 code review 检查加锁点）
- 门面无子对象指针泄漏（grep 验证：对外头文件不含 CameraControl/MCUDriver 类型）
- AppContext 装配收拢验证：`AppContext.cpp` 设备相关 ≤ 3 行（构造+open+close）
