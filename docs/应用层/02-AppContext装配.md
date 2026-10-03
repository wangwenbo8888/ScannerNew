# 应用层 02 · AppContext 装配 协作文档

> **版本**：v1 ｜ **更新日期**：2026-10-01 ｜ **基于代码**：1f2b19a（2026-09-27）
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

- **模块定位**：`app/AppContext.h`＋`app/AppContext.cpp`（227/1125 行）＝ 应用组合根——创建并持有全部运行时组件（Infra/Data/Service/HAL/Workflow 五类，物理散布于 `base/`＋`modules/`＋`app/`），统一注入经 `WorkflowContext`；并承担命令通道点火、扫描会话生命周期、自检/故障桥/设备灯联动、帧流分路、中段模拟提取
- **上下游依赖**：
  - 上游：`app/main.cpp`（initialize/shutdown/startDevicesAsync 调用方）；UI（MainWindow/ScannerWindow 经 getter 与会话点火口）
  - 下游：base（EventBus/types）；模块06（四容器＋ParameterManager＋file_io）；模块10（StateMachine/FaultHandler/CommandGate/PerfMonitor/ObsLogger）；模块08（DeviceManager/HardwareMonitor/SelfCheckCollector/CameraFactory）；模块07（ISceneFeed 契约＋SimScanSource＋CpuTopology）；模块01/02/04 工作流
- **关键约束**：
  - 08 不链 10、02/01/04 不依赖 10——跨模块接线只能在本组合根完成
  - EventBus 同步分发持总线锁：订阅回调内禁止 transition/publish/直写 UI（故障桥经 QTimer 延后主线程）
  - 工作流点火 handler 须毫秒级返回——装配/GBA 终局批一律后台线程
  - shutdown 只允许执行一轮（once 守卫；实测无守卫时退出链跑 3+ 轮致进程挂死）

### 1.2 人的补充与需求〔人填写，AI 不得覆盖〕

> 人每次迭代在此追加：新需求 / 问题反馈 / 背景补充。**注明日期**，旧内容不删除。

- （YYYY-MM-DD）{{人写的内容}}

---

## 2. 设计目标〔人定，AI 整理〕

| 编号 | 要实现的功能 | 实现方法 | 量化标准 | 状态 |
|------|------------|---------|---------|------|
| G1 | 组合根唯一装配点 | 全部组件在 AppContext 统一构造＋unique_ptr 持有，经 getter/WorkflowContext 暴露非所有权指针 | 组件 18 个；生命周期＝整个应用；成员声明序＝逆析构序 | 已达成 |
| G2 | 分层装配与对称关闭 | Infra→Data→Service→HAL→WorkflowContext→Workflow 顺序装配；shutdown 逆序＋once 守卫 | 关闭链恰好执行一轮（atomic 守卫） | 已达成 |
| G3 | 命令通道统一点火 | 标定/扫描/后处理经 CommandGate（pre 谓词＋handler），完成经工作流 onFinished 回调合账 | 命令全集 7 条注册（4 条由 app 填 handler/pre）；handler 毫秒级返回〔待确认：≤100ms〕 | 已达成 |
| G4 | 扫描会话全生命周期 | arm（备会话不启采）/start（直启兼容）/stop（GBA 后台合账）/pause/resume（就绪态保活）统一入口 | 会话代守卫防停启竞态过期回滚；pause 保活融合云/obs 账本 | 已达成 |
| G5 | 自检与状态联动 | 6 项自检→system_ready（S1→S2 唯一出口）；故障桥 Error→S7 安全停机＋红灯；切态随发设备指示灯 | 自检项 6；S7 链含 toIdle＋红灯 | 已达成 |
| G6 | 帧流分路与渲染推送 | 相机帧回调双投递（预览 FrameBuffer＋扫描会话环）＋调试分路 tap；SceneFeedAdapter 把流水线云快照 queued 推 UI | tap 心跳每 300 帧记日志 | 已达成 |
| G7 | 配置外置 | camera.json（相机装机口径）＋calibration.json（合并 laser_calib.json）＋device_params.txt（参数档 IO） | 缺档用内置默认并留痕，不阻断启动 | 已达成 |
| G8 | 中段模拟提取（调试件） | UI 开关→下个会话由 SimScanSource 替换提取观测，配准/融合/渲染走生产代码 | 默认关＝零影响；激光主源 pointcloud_100M.ply 前 3000 万点 | 已达成 |

---

## 3. 目标与现状的差距〔AI〕

| 编号 | 对应目标 | 差距描述 | 拟议方向 |
|------|---------|---------|---------|
| D1 | G5 | license 自检项占位恒 true——加密狗到货后须接 USB 枚举/厂商 SDK 实检 | 狗到货后替换 |
| D2 | G7 | config 三档均为相对路径（依赖工作目录＝exe 目录；VS 调试靠 VS_DEBUGGER_WORKING_DIRECTORY 兜底） | 启动时按 exe 目录拼绝对路径 |
| D3 | G3 | start_scan 的 pre 谓词只查仓库整档 readyForScan；06 逐温档 K/D＋激光温度表出口查表接入后判据应升级 | 随 06 出口查表落地升级 |
| D4 | G4 | canEnterEditSession 未检查激光云非空（现查标志点/点云总计数） | P4 接 ILaserFuse 出口后补 |
| D5 | G6 | SceneFeedAdapter::pushPostureView 为空实现（A 姿态实时视图待 01 标定接线期落地） | 01 deps.sceneFeed 接线时实现 |
| D6 | G1/G2 | AppContext.cpp 单文件 1125 行（装配＋命令注册＋会话点火＋模拟源组装堆一处），全系统拓扑审查面大 | 按功能域拆文件〔待确认〕 |

---

## 4. 已有代码现状〔AI〕

### 4.1 总图（ASCII 箱线风格，≤2 张）

```
【装配主链（initialize 顺序）】
EventBus ──► Data 四容器 ──► Service 五件 ──► HAL 三件 ──► WorkflowContext ──► 三工作流
（Infra）     （06）          （10/06）       （08 门面）    （app 聚合器）     （02/01/04）
                                │
                                └ CommandGate 全集 7 命令（4 条由 app 填点火） ◄── onFinished 回合账
HAL 接线：DeviceManager（门禁回调→StateMachine）· HardwareMonitor（→DeviceStateCache/EventBus，
          周期 1s）· 相机经 08 工厂构造（装机口径 camera.json）
关闭：shutdown 逆序＋once 守卫（join 三后台线程→巡检停→退订→工作流停→故障停→门面 close）
```

### 4.2 功能实现清单（按功能域分组）

#### 4.2.1 组件拥有与生命周期（支撑域，服务全部目标）

| 功能 | 对应文件 | 实现方法一句话 | 达成情况 |
|------|---------|--------------|---------|
| Infra/Data 层持有 | `app/AppContext.h` | EventBus（`base/EventBus.h`）；FrameBuffer(60)/PointCloudBuffer/DeviceStateCache（06）；CalibrationRepository（06，load config/calibration.json 合并 laser_calib.json） | G1：已达成 |
| Service 层持有 | `app/AppContext.h` ＋ `modules/10_observability/StateMachine.h` 等 | StateMachine/FaultHandler/CommandGate/PerfMonitor（10）＋ParameterManager（06） | G1：已达成 |
| HAL 层持有 | `app/AppContext.h` ＋ `modules/08_devicemgmt/DeviceManager.h` 等 | DeviceManager（相机＋MCU 门面）＋SelfCheckCollector＋HardwareMonitor；相机经 `modules/08_devicemgmt/CameraFactory.h` 工厂构造 | G1：已达成 |
| 渲染推送/工作流持有 | `app/AppContext.h` ＋ `app/SceneFeedAdapter.h` ＋ `app/WorkflowContext.h` | SceneFeedAdapter（app）＋WorkflowContext（app）＋Scan/Calibration/PostProcess Workflow（02/01/04） | G1：已达成 |
| 析构序设计 | `app/AppContext.h` | 成员声明序＝逆析构序（gate/perf 先亡、sceneFeed 先于总线亡、门面最后亡） | G1/G2：已达成 |
| shutdown 逆序 | `app/AppContext.cpp` | once 守卫→join 装配/完成/设备三线程→巡检停→退订故障桥/灯桥→三工作流 stop→faultHandler stop→门面 close | G2：已达成 |

#### 4.2.2 命令通道与会话点火（服务 G3/G4）

```
【扫描会话（就绪流程口径）】
UI 模式键 ──► armScanSession（设备就绪查＋帧流双投递＋setCaptureMode）
                ▼
        gate.submit("start_scan") ──拒──► 带因返回（pre：readyForScan 门禁）
                │过                          │失败·会话代未变
                ▼                            └─► 设备收口＋notifyCompleted(false)＋UI 复原
        后台装配线程（仓库查表→sim 开关→工作流 initialize＋start）
                ▼
        设备 M 键开扫（N10 四管掩码）→ 帧入会话环
                ▼
UI 点键关闭 ──► stopScanSession（stopCapture N11H0 → finish_scan 后台：stop＋GBA 终局）
                ▼
        notifyCompleted 合账 ＋ 标志点落库
```

| 功能 | 对应文件 | 实现方法一句话 | 达成情况 |
|------|---------|--------------|---------|
| 命令注册 | `app/AppContext.cpp` ＋ `modules/10_observability/DefaultCommands.h` | DefaultCommands 全集 7 条注册；app 为其中 4 条填 handler/pre（start_calibration/start_scan/finish_scan/start_postprocess） | G3：已达成 |
| start_scan 后台装配 | `app/AppContext.cpp` | scanStartThread_ 后台装配＋会话代 scanActGen_（过期回滚跳过，防停启竞态扑杀新会话） | G3/G4：已达成 |
| finish_scan 后台终局 | `app/AppContext.cpp` | finishThread_ 执行 stop()（含 GBA 终局遍，分钟级）＋notifyCompleted 合账 | G3/G4：已达成 |
| 就绪/直启/停止 | `app/AppContext.cpp`（armScanSession/startScanSession/stopScanSession） | arm＝备会话不启采（M 键才开扫）；start＝arm＋startCapture；stop＝N11H0＋finish_scan | G4：已达成 |
| 就绪态保活 | `app/AppContext.cpp`（pause/resume/isScanSessionPaused） | 停采集保活会话（融合云/obs 账本保留）；续采 N10 参数＋N11H1 重启触发 | G4：已达成 |
| 编辑门禁 | `app/AppContext.cpp`（canEnterEditSession） | （S2 待机或暂停就绪态）且（标志点或点云＞0）——05/03 编辑入口的唯一事实源 | G4：部分达成（D4） |
| 后处理点火 | `app/AppContext.cpp`（startPostProcessSession） | 设输出路径/跳段位→gate submit start_postprocess（pre 拦空点云；批算在 04 postThread_） | G3：已达成 |

#### 4.2.3 自检·故障桥·状态联动（服务 G5）

| 功能 | 对应文件 | 实现方法一句话 | 达成情况 |
|------|---------|--------------|---------|
| 自检清单 | `app/AppContext.h`（SelfCheck） | 6 项（camera/serialPort/mcuLink/bgLight/laser/license）mutex 防护＋快照口 | G5：已达成（license 占位 D1） |
| S1→S2 出口 | `app/AppContext.cpp`（notifySelfCheckItem） | 全过且处 Init→submit("system_ready")（恰一次，重复幂等失败） | G5：已达成 |
| 设备后台启动 | `app/AppContext.cpp`（startDevicesAsync） | CPU 拓扑记录→DeviceManager open→serialPort/license 上报→Camera/MCU 状态落缓存→startupSelfCheck 余四项 | G5：已达成 |
| 08→10 故障桥 | `app/AppContext.cpp`（faultBridgeSubId_ 订阅） | FaultOccurred（Error 级）→QTimer 延后主线程（避总线锁重入）→转 S7＋toIdle＋红灯 | G5：已达成 |
| 设备指示灯 | `app/AppContext.cpp`（ledSubId_ 订阅） | StateChanged→N14 S1-S4（黄/红/绿/蓝）；open 成功补发当前态；去重归 DeviceManager | G5：已达成 |

#### 4.2.4 帧流分路与渲染推送（服务 G6）

| 功能 | 对应文件 | 实现方法一句话 | 达成情况 |
|------|---------|--------------|---------|
| 帧流双投递 | `app/AppContext.cpp`（armScanSession 内 startFrameStream） | ①预览链→06 FrameBuffer；②扫描链→pushSessionFrame（带温度/模式/T-V 判定随帧） | G6：已达成 |
| 调试分路 | `app/AppContext.h`（setDebugFrameTap） | 相机 SDK 线程直调；每 300 帧记 tap 心跳/空转双路日志（监视窗死因定位锚） | G6：已达成 |
| 渲染推送适配器 | `app/SceneFeedAdapter.cpp` ＋ `modules/07_pipelinemgmt/pipelines/ISceneFeed.h` | ISceneFeed 首个实现：pushCloudSnapshot 调用线程值拷贝→queued 信号推 UI；冻结期丢弃保末帧；latestMarkers 末次快照缓存 | G6：已达成（D5） |
| 依赖聚合器 | `app/WorkflowContext.cpp` | 窄接口注入工作流（Data 四件/Service 两件/EventBus/SceneFeed）＋publishProgress/publishEvent 快捷口 | G6：已达成 |

#### 4.2.5 配置装载（服务 G7）

| 功能 | 对应文件 | 实现方法一句话 | 达成情况 |
|------|---------|--------------|---------|
| 相机装机口径 | `app/AppContext.cpp`（camera.json 解析） | L/R 设备号、右图 180°、触发源、previewFps、帧号严格配对、时间戳配对；缺档默认＋WARN | G7：已达成 |
| 标定档 | `app/AppContext.cpp` ＋ `modules/06_fileio/CalibrationRepository.h` | config/calibration.json＋同目录 laser_calib.json 工厂档自动合并；未装载不阻断 | G7：已达成 |
| 设备参数档 | `app/AppContext.cpp`（ParamIo 注入 DeviceManager） | load/persist 落 config/device_params.txt（防抖 2s＋close 兜底归 08 管理） | G7：已达成 |

#### 4.2.6 中段模拟提取（服务 G8，调试件）

| 功能 | 对应文件 | 实现方法一句话 | 达成情况 |
|------|---------|--------------|---------|
| 开关 | `app/AppContext.h`＋`app/AppContext.cpp`（setSimExtract/assembleSimSource） | UI「模拟数据」开关，start_scan 装配时读取组装；默认关零影响 | G8：已达成 |
| 数据源组装 | `app/AppContext.cpp`（assembleSimSource） | 标志点←PLY（JMW_SIM_MARKERS_PLY 覆写，缺省 D:/markers_30.ply）；激光←导入 stash→JMW_SIM_LASER_PLY 覆写→D:/pointcloud_100M.ply 前 3000 万点→builtin 大平面多级兜底 | G8：已达成 |
| 铺展与轨迹 | `app/AppContext.cpp` ＋ `modules/07_pipelinemgmt/pipelines/scan/SimScanSource.h` | FPS 最远点采样铺标志点（工作距护栏＋离群围栏）；SimTrajParams 递增观测调度（设备静止） | G8：已达成 |
| 限量读取器 | `app/AppContext.cpp`（匿名命名空间两个限量读取器） | 装配层自持大文件读取（06 importPLY 对 3100 万点级实测失败回退） | G8：已达成 |

### 4.3 有雏形但未完成的部分

- pushPostureView 空实现（D5）
- canEnterEditSession 激光云非空检查（D4）
- license 加密狗实检（D1）

---

## 5. 修改记录〔AI〕

| 日期 | 类型 | 内容 | 原因 |
|------|------|------|------|
| YYYY-MM-DD | 代码 / 目标 / 文档 | {{改了什么}} | {{为什么改}} |
