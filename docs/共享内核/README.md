# 共享内核 协作文档

> **版本**：v1 ｜ **更新日期**：2026-10-01 ｜ **基于代码**：1f2b19a
>
> **维护规则**（多轮迭代必读）：
>
> - 标注〔人〕的内容由人填写，**AI 不得改写**，只允许在其后追加
> - 标注〔AI〕的内容由 AI 维护，随代码变更刷新
> - 正文永远只保留**最新状态**；被修改/废弃的历史一律移入第 5 节存底
>
> **排版三铁律**：
>
> 1. 每个要点、子要点单独一行，禁止一行塞多个要点
> 2. 编号连续清晰：目标用 G1/G2…，差距用 D1/D2…，全文贯通引用
> 3. 流程图/架构图用 ASCII 箱线风格：主流程从上往下连线，分支从箱体右侧引出；全文档图 ≤5 张，图内单行 ≤100 字符，总量 ≤250 行

---

## 1. 信息〔人＋AI〕

### 1.1 模块背景〔AI 预填，迭代时刷新〕

- **模块定位**：全工程最底层的「公共零件箱」——集中供给公共类型＋事件通知总线；各层都可依赖它，它不依赖任何人（仅 C++ 标准库）
- **上下游依赖**：
  - 上游：无（仅 C++ 标准库；EventBus 实例由 `app/AppContext.cpp` 创建并注入各模块）
  - 下游：05/06/07/08/10 五个 mod_* 库＋LeadScannerK2；09 算子库不依赖 base
    - 发布方：02 ScanWorkflow、07 EventBusEventSink、08 DeviceManager、10 StateMachine/FaultHandler/CommandGate/PerfMonitor
    - 订阅方：10 FaultHandler、app 故障桥/LED 桥/UI 状态图标
- **关键约束**：
  - 只依赖 C++ 标准库，OpenCV/CUDA/Qt 等第三方一律禁入
  - 禁业务代码，只放公共类型＋事件总线
  - 命名空间：类型用 `Scanner::`，事件总线用 `Scanner::infra::`
  - 事件只传通知（控制/状态），不搬数据载荷
  - `UserDefined` 是全订阅通配哨兵：不得用 `subscribe(EventType::UserDefined, …)` 表达「只订自定义事件」——那等于订阅了所有事件
  - `publish` 同步派发且全程持锁（非递归锁）：回调内禁止再调同一总线的 publish/subscribe/unsubscribe，否则死锁
  - 慢回调会阻塞其他线程的总线操作（订阅/发布/退订共用一把锁）

### 1.2 人的补充与需求〔人填写，AI 不得覆盖〕

> 人每次迭代在此追加：新需求 / 问题反馈 / 背景补充。**注明日期**，旧内容不删除。

- （YYYY-MM-DD）{{人写的内容}}

---

## 2. 设计目标〔人定，AI 整理〕

| 编号 | 要实现的功能 | 实现方法 | 量化标准 | 状态 |
|------|------------|---------|---------|------|
| G1 | 全工程统一公共类型 | types.h 单头集中定义：工厂方法＋强类型枚举＋工程别名 | 覆盖 14 个公共类型〔待确认〕；第三方依赖 = 0 | 已达成 |
| G2 | 线程安全事件通知总线 | EventBus 订阅/发布，互斥保护＋同步派发，UserDefined 通配 | 覆盖 22 种 EventType〔待确认〕；订阅/发布/退订全程线程安全 | 已达成 |
| G3 | 极薄构建接入 | 单静态库 base，纯头类型＋单编译单元 | 编译单元 = 1；目录文件 = 4〔待确认〕；05/06/07/08/10 五个 mod_* 库＋LeadScannerK2 可直接链接 | 已达成 |

---

## 3. 目标与现状的差距〔AI〕

| 编号 | 对应目标 | 差距描述 | 拟议方向 |
|------|---------|---------|---------|
| D1 | G1 | EventType/DeviceState/ScanMode 计划拆出到 07/06/02（types.h:6-7 注释声明），至今仍合一 | 按 Phase 计划拆分，或正式关闭该计划〔待确认〕 |
| D2 | G2 | publishSync 定位「关键通道专用」，实现与 publish 完全等同（EventBus.cpp:16-18），语义未落地 | 区分实现（独立通道/优先级），或删冗余接口〔待确认〕 |
| D3 | G2 | 派发全程持锁＋同步执行：慢回调阻塞其他线程；回调内重入调总线即死锁（非递归锁） | 改快照派发（锁内拷贝订阅表、锁外执行回调），或固化为永久约束〔待确认〕 |
| D4 | G2 | base 无专属单元测试，仅经 07/08/10 测试间接覆盖使用路径 | 补 base 专属测试：订阅/通配/退订/清空/计数〔待确认〕 |

---

## 4. 已有代码现状〔AI〕

### 4.1 总图（ASCII 箱线风格，≤2 张）

```
【base 定位：全工程最底层公共零件箱】
┌────────────────────────────────────┐
│ base（STATIC 静态库）               │
│  types.h  ──公共类型──► 全部下游     │
│  EventBus ──事件通知──► 全部下游     │
└─────────────────┬──────────────────┘
                  ▼ 被链接
   05 / 06 / 07 / 08 / 10（mod_* 库）
                  ▼
        app / LeadScannerK2（01-04 编入）
```

### 4.2 功能实现清单（按功能域分组）

#### 4.2.1 公共类型供给（服务 G1）——一个头文件集中供给全部公共类型

静态清点型域，仅表格：

| 功能 | 对应文件 | 实现方法一句话 | 达成情况 |
|------|---------|--------------|---------|
| 统一执行结果 | `base/types.h` | Result 四工厂（ok / fail / degraded / warning；fail 另有单参重载，错误码 -1）＋四字段＋isDegraded 等判定 | G1：已达成 |
| 质量与等级枚举 | `base/types.h` | QualityFlag 四值；FaultSeverity 四值；ContractLevel 三值 | G1：已达成 |
| 位姿与帧标识 | `base/types.h` | Pose（R[9]＋t[3]＋帧号＋时间戳＋identity()）；FrameId/TimestampMs 别名 | G1：已达成 |
| 设备与扫描枚举 | `base/types.h` | DeviceState 四值；ScanMode 四值（MarkerOnly / MarkerPlusLaser / FineScan / DeepHoleScan） | G1：已达成 |
| 事件类型与结构 | `base/types.h` | EventType 22 值（含 mod10 可观测扩展 9 项）；Event 五字段（type/sourceId/timestamp/param1/param2） | G1：已达成 |
| 健康指标快照 | `base/types.h` | HealthMetrics 10 字段（CPU/GPU/内存/磁盘/帧率/丢帧率/时间戳，-1=未采） | G1：已达成 |
| 智能指针别名 | `base/types.h` | UPtr/SPtr＝unique_ptr/shared_ptr 的工程统一别名 | G1：已达成 |

#### 4.2.2 事件总线（服务 G2）——线程安全的订阅/发布通知通道

```
publish(event)
    │ 加互斥锁
    ▼
┌────────────────────────┐
│ 遍历订阅表 subscribers_ │──type==UserDefined──► 通配：按全订阅派发
└───────────┬────────────┘
            ▼ type==event.type
   匹配订阅者回调（同步执行，锁内）
            ├──回调内再调同总线──► 死锁（非递归锁，禁止）
            ▼
      返回 Result::ok()
```

| 功能 | 对应文件 | 实现方法一句话 | 达成情况 |
|------|---------|--------------|---------|
| 订阅管理 | `base/EventBus.h` / `base/EventBus.cpp` | subscribe / subscribeAll（内部即 subscribe(UserDefined)）/ unsubscribe / clear；nextId_ 原子自增发号 | G2：已达成 |
| 同步派发 | `base/EventBus.cpp` | publish 持锁遍历，UserDefined 或 type 匹配即同步回调 | G2：已达成 |
| 关键通道 | `base/EventBus.cpp` | publishSync 直接转调 publish（EventBus.cpp:16-18），语义未区分 | G2：部分达成（D2） |
| 观测辅助 | `base/EventBus.cpp` | getSubscriberCount 持锁返回订阅总数 | G2：已达成 |

#### 4.2.3 构建接入（服务 G3）——单静态库＋双路径 include 解析

静态清点型域，仅表格：

| 功能 | 对应文件 | 实现方法一句话 | 达成情况 |
|------|---------|--------------|---------|
| 静态库 | `base/CMakeLists.txt` | add_library(base STATIC EventBus.cpp)；types.h 纯头不入编译 | G3：已达成 |
| include 解析 | `CMakeLists.txt`（根，146 行）＋ `base/CMakeLists.txt` | 根全局 include 支持 `#include "base/xxx.h"` 前缀写法；库 PUBLIC 导出 base/ 目录，不带前缀写法亦可解析 | G3：已达成 |

### 4.3 有雏形但未完成的部分

- publishSync「关键通道」语义：接口已留位，实现与 publish 等同（见 D2）
- EventType/DeviceState/ScanMode 拆出到 07/06/02：仅头注释声明，未执行（见 D1）

---

## 5. 修改记录〔AI〕

| 日期 | 类型 | 内容 | 原因 |
|------|------|------|------|
| YYYY-MM-DD | 代码 / 目标 / 文档 | {{改了什么}} | {{为什么改}} |
