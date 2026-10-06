# 应用层 03 · MainWindow 主窗口 UI 协作文档

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

- **模块定位**：`app/MainWindow.h`＋`app/MainWindow.cpp`（176/2714 行）＝ 无边框全屏主窗口 "LeadScan K2"——全部 UI 布局、扫描/标定/文件/编辑操作入口、渲染接线与系统信息可视化；持 `AppContext*` 按需取用，不 new 组件
- **上下游依赖**：
  - 上游：`app/main.cpp`（构造）；AppContext（会话点火/仓库/EventBus/SceneFeed，见 02）
  - 下游：模块03 OSGWidget（渲染）；模块02/04 工作流（编辑物理化/进度回调）；模块07 契约（融合云移除/obs 剔除——经 02 工作流出口触达）；模块06 file_io（导入导出）＋仓库；模块08 DeviceManager（参数账本）；模块10 StateMachine/PerfMonitor；app/stubs 桩（标定槽）
- **关键约束**：
  - EventBus 同步分发持总线锁——订阅回调只拷贝值再 queued 切主线程（对齐 02 红线）
  - 激光云全域扫描达数百万点——显示/导出须抽样与节流，不得拖死 UI 线程
  - 编辑门禁唯一事实源＝AppContext::canEnterEditSession，UI 不得自设判据

### 1.2 人的补充与需求〔人填写，AI 不得覆盖〕

> 人每次迭代在此追加：新需求 / 问题反馈 / 背景补充。**注明日期**，旧内容不删除。

- （YYYY-MM-DD）{{人写的内容}}

---

## 2. 设计目标〔人定，AI 整理〕

| 编号 | 要实现的功能 | 实现方法 | 量化标准 | 状态 |
|------|------------|---------|---------|------|
| G1 | 主窗口框架 | 无边框全屏，五区布局（标题/导航/工具栏/左面板/3D 区）＋悬浮工具条 | 布局区 5＋悬浮条 1；三态 SVG 图标体系（黑/红/灰） | 已达成 |
| G2 | 扫描会话操作闭环 | 四模式键（标点/面片/精细/深孔）arm 就绪→设备 M 键开扫→点键关闭＋GBA；navBar「扫描」＝面片直启/停止；参数预设与三参比例联动 | 就绪弹窗 200ms 轮询自动关；活跃键红框「停止扫描」 | 已达成 |
| G3 | 点云渲染与工程树 | SceneFeed queued 信号直渲（标志点定向圆盘/激光）＋工程树实时计数＋节流快照导出 | 显示抽样上限约 150 万点；快照每 30 拍写盘 | 已达成 |
| G4 | 文件导入导出 | 数据管理菜单：导入标志点/点云/网格/工程；导出标志点/点云；后处理导出 STL（真点火）；工程文件 | 导入 4 项/导出 4 项；工程文件 2 项未实现 | 进行中（差距 D2/D5） |
| G5 | 编辑工具链 | 三栏选择模型＋套索/多段线删除＋Ctrl+Z＋编辑物理化（融合云＋obs 剔除）＋门禁 | 圈选结束三栏复位；删除随会话关闭物理化 | 已达成 |
| G6 | 系统信息与状态可视化 | 六卡 1s 刷新（CPU=PDH Utility 口径）＋自检横幅＋状态栏 7 态常驻指示＋全局优化进度弹窗 | 刷新 1s；7 态文案/配色映射 | 已达成 |
| G7 | 标定 UI | 校准设备→CalibDialog＋标定分屏（STL 锁定视角＋2D 标定板＋彩条滑块）→采集链 | 相机标定 15 帧采集口径 | 进行中（差距 D1） |
| G8 | 相机预览监视（调试件） | 扫描启动弹出左右灰度图＋T/V 组/帧号/偏移/帧率信息行 | previewFps 节流（camera.json）；降采样 1/4 | 已达成 |

---

## 3. 目标与现状的差距〔AI〕

| 编号 | 对应目标 | 差距描述 | 拟议方向 |
|------|---------|---------|---------|
| D1 | G7 | 标定槽走 stubs 空桩：`getCameraControl()` 返回 nullptr→相机检查恒失败；CameraCalibWorkflow.run 返回假成功（"Stub"）。真实链（01 CalibrationWorkflow＋start_calibration 命令）已具备但 UI 未接 | 改接 AppContext 统一标定入口 |
| D2 | G4 | 硬编码路径 4 处：`D:/pointcloud_100M.ply`、`E:/workfold/…calib_debug.log`、`…JEAMMSCAN.stl`、`…import_debug.log`（后三者调试直写文件） | 收敛到配置/工程目录 |
| D3 | G1 | dark.qss 未登记 qrc，样式不生效（同 01-D1） | 登记 qrc |
| D4 | G5 | 激光侧删除物理化仅在 JMW_BUILD_CUDA 宏下接 laserFuse；非 CUDA 构建为空操作 | 补 CPU 出口或明示限制 |
| D5 | G4 | 工程文件导入/导出未实现（仅占位提示，序列化格式规划中） | 格式定案后补 |
| D6 | G3 | MainWindow.cpp 单文件 2714 行（UI 构建/槽逻辑/信息面板/编辑接线堆一处） | 按布局/槽/信息面板拆分〔待确认〕 |
| D7 | G7 | m_integrateTestDialog 声明为 QWidget* 却 5 处 static_cast\<LEADSCANSeries*\>（对话框存在时为未定义行为）；m_series 成员从未赋值使用（死成员） | 随 D1 桩退役一并清理 |

---

## 4. 已有代码现状〔AI〕

### 4.1 总图（ASCII 箱线风格，≤2 张）

```
【MainWindow 布局树】
QMainWindow（无边框 "LeadScan K2"，全屏）
├─ 标题栏 createTitleBar（logo/翻页三钮/工程名/保存/最小化/关闭）
├─ 导航栏 createNavBar（菜单/扫描/管理/集成测试/加载点云 · 右侧五模式展示键）
├─ 工具栏 createToolBar（数据管理/校准设备/四扫描模式键/正反/切面/重置 · 模拟数据开关）
├─ 内容行（stretch 2:5）
│   ├─ 左面板 createLeftPanel（项目树 / 参数面板 / 系统信息）
│   └─ 3D 区 create3DViewArea（远近滑条＋OSGWidget——模块03）
└─ 悬浮工具条（底部三栏选择模型＋操作组，eventFilter 拖动/强制 tip）
状态栏：自检横幅 / 消息 / 右下角 7 态常驻指示●
```

### 4.2 功能实现清单（按功能域分组）

#### 4.2.1 窗口骨架与图标系统（支撑域，服务全部目标）

| 功能 | 对应文件 | 实现方法一句话 | 达成情况 |
|------|---------|--------------|---------|
| 布局组装 | `app/MainWindow.cpp`（构造＋createXxx 系列） | 五区＋悬浮条组装；resize 时悬浮条随 3D 区底边居中 | G1：已达成 |
| SVG 图标 | `app/MainWindow.cpp` ＋ `app/resources.qrc` | renderSvg 按 dpr 渲染 QPixmap；按钮工厂统一生成 | G1：已达成（dark.qss 未登记 D3） |
| 自定义滑条 | `app/MainWindow.cpp`（ArrowSlider） | 渐变槽 pixmap＋箭头手柄绘制（远近/左右/前后彩条复用） | G1：已达成 |
| 悬浮条交互 | `app/MainWindow.cpp`（eventFilter） | 左键拖动窗体；无边框半透明窗下 HoverEnter 强制弹 QToolTip | G1：已达成 |

#### 4.2.2 扫描会话操作（服务 G2/G8）

```
【四模式键状态流（idx 2 标点/3 面片/4 精细/5 深孔）】
点击模式键
   ▼
会话活跃？ ──是──► 编辑成果物理化 → stopScanSession（后台 GBA＋合账）
   │                └─ 本键＝纯关闭；他键＝落下继续启新模式
   ▼ 否
套用预设（标点/面片/参数压账本）→ armScanSession（备会话不启采）
   ▼
本键红框「停止扫描」＋弹就绪窗（200ms 轮询 isCapturing 自动关/取消终止）
   ▼
设备 M 键开扫（提示「按 M 停止」）
```

| 功能 | 对应文件 | 实现方法一句话 | 达成情况 |
|------|---------|--------------|---------|
| 四模式键 | `app/MainWindow.cpp`（createToolBar 扫描键 lambda） | 关闭会话→编辑物理化→预设→arm；工程树首会话建「标记点 001」节点 | G2：已达成 |
| navBar 扫描 | `app/MainWindow.cpp`（onScanClicked） | 活跃则停（stopScanSession）；空闲则面片直启（startScanSession）＋弹相机监视 | G2：已达成 |
| 就绪窗口 | `app/MainWindow.cpp`（showScanReadyPrompt） | 「按设备 M 键开始扫描」；200ms 轮询 isCapturing 自动关；取消＝终止会话 | G2：已达成 |
| 按钮态视觉 | `app/MainWindow.cpp`（setScanButtonVisual） | 活跃＝红框红字「停止扫描」，停止复原四模式名 | G2：已达成 |
| 参数联动与预设 | `app/MainWindow.cpp`（applyMarkerPreset 等） | 参数1 旋钮 1-100 比例映射曝光 1-5ms/补光 0-100/激光 0-100（经 08 账本）；标点/面片各带推荐预设 | G2：已达成 |
| 相机预览监视 | `app/MainWindow.cpp`（showCameraMonitor） | 调试帧分路（tap）→降采样 1/4→queued 刷图；信息行含 T/V 组/帧号/有符号偏移/接收与流水线帧率 | G8：已达成 |

#### 4.2.3 渲染接线与工程树（服务 G3）

| 功能 | 对应文件 | 实现方法一句话 | 达成情况 |
|------|---------|--------------|---------|
| 标志点直渲 | `app/MainWindow.cpp`（markerCloudUpdated 接线） | queued 收点→工程树计数（每 30 拍）→setMarkers 落仓库→定向圆盘上屏；markers_snapshot.ply 节流导出 | G3：已达成 |
| 激光直渲 | `app/MainWindow.cpp`（laserCloudUpdated 接线） | 超过 150 万点按步长抽样上屏；laser_snapshot.ply 每 30 拍导出；数据通道已退役归 07 直写仓库 | G3：已达成 |
| 渲染事件桥 | `app/MainWindow.cpp`（setFaultSink） | OSG 渲染事件→EventBus FaultOccurred（sourceId=0x03，Degraded→Warning） | G3：已达成 |
| 仓库快照重载 | `app/MainWindow.cpp`（startInfoTimer 内 cloudTimer） | 500ms 拉 PointCloudBuffer 快照→非扫描期重载 3D（扫描期激光走直推不双份） | G3：已达成 |

#### 4.2.4 数据管理菜单（服务 G4）

| 功能 | 对应文件 | 实现方法一句话 | 达成情况 |
|------|---------|--------------|---------|
| 导入 | `app/MainWindow.cpp`（数据管理菜单） | 标志点/点云（同时 stash 供模拟源）/网格/工程四项；点云走 `modules/06_datamgmt/file_io.h` | G4：已达成（工程占位 D5） |
| 导出 | 同上 | 标志点/点云走仓库内存直导（数量回显）；后处理导出 STL＝真点火 | G4：已达成 |
| 后处理入口 | `startPostProcessSession` 调用＋进度回调 | gate 点火 S2→S6；阶段名/百分位透传状态栏 | G4：已达成 |

#### 4.2.5 编辑工具链（服务 G5）

| 功能 | 对应文件 | 实现方法一句话 | 达成情况 |
|------|---------|--------------|---------|
| 三栏选择模型 | `app/MainWindow.cpp`（createBottomToolBar） | 对象类型/选择类型/工具类型三互斥组＋操作组（隐藏待 P4）；checked 切 (3) 号图标 | G5：已达成 |
| 套索/多段线 | `app/MainWindow.cpp`（选择键接线） | ensureEditAllowed 门禁→OSGWidget 圈选删除模式；圈选结束三栏复位＋统计刷新 | G5：已达成 |
| 编辑物理化 | 四模式键 lambda（materializeEdits） | 会话关闭前把显示级删除路由到真账本：markerFuse.removePoints＋obsAccumulator.excludeMarkerObs；激光侧走 laserFuse（CUDA 宏） | G5：部分达成（D4） |
| 撤销 | `app/MainWindow.cpp`（Ctrl+Z 接线） | OSGWidget undoDelete＋可见标志点计数同步 | G5：已达成 |

#### 4.2.6 系统信息与状态可视化（服务 G6）

| 功能 | 对应文件 | 实现方法一句话 | 达成情况 |
|------|---------|--------------|---------|
| 信息六卡 | `app/MainWindow.cpp`（updateInfoSection） | 1s：连接（DeviceStateCache）/点云数/实测帧率/MCU 温度（G02 四路取最高）/CPU（PDH Utility 钳位）/内存 | G6：已达成 |
| PerfMonitor 驱动 | `app/MainWindow.cpp`（updateInfoSection 头部） | 挂 m_infoTimer 每 1s poll()（provider←08 HardwareMonitor 快照，见 02） | G6：已达成 |
| 自检横幅 | `app/MainWindow.cpp`（updateInfoSection） | 1s 轮询 selfCheckSnapshot 拼 ✓/✗ 文案；完成后再显示 10s 撤 | G6：已达成 |
| 7 态指示 | `app/MainWindow.cpp`（updateStateIndicator 等） | 订阅 StateChanged（锁内只拷贝值，queued 刷状态栏右下角色点＋文案） | G6：已达成 |
| GBA 进度弹窗 | `app/MainWindow.cpp`（finalBAProgress 接线） | finish 后台全局优化的百分位/阶段文案→模态进度弹窗，100% 自动关 | G6：已达成 |

#### 4.2.7 标定 UI（服务 G7）

| 功能 | 对应文件 | 实现方法一句话 | 达成情况 |
|------|---------|--------------|---------|
| 标定对话框 | `modules/01_calibration/CalibDialog.h` 接线 | cameraCalibClicked/laserCalibClicked 两信号→采集链 | G7：已达成（链为桩 D1） |
| 标定分屏 | `modules/01_calibration/CalibDisplay.h` | STL 扫描仪场景锁定视角＋2D 标定板＋左右/前后彩条滑条 | G7：已达成 |
| 切回默认 | `app/MainWindow.cpp`（扫描四模式键/正反键接线） | 标定板可见时才恢复：隐藏分屏→清场景→显悬浮条 | G7：已达成 |

### 4.3 有雏形但未完成的部分

- 标定槽的桩链（`app/stubs/LEADSCANSeries.h` 等 5 桩）——真实 01 工作流接线后整套退役（D1/D7）
- 激光线宽度测量（预览监视信息行占位 0——实现暂停待排障结论，原实现见 git 历史）
- 切面扫描/重置项目两工具键为展示位，未接任何槽（正反键也仅保留切回默认行为）

---

## 5. 修改记录〔AI〕

| 日期 | 类型 | 内容 | 原因 |
|------|------|------|------|
| YYYY-MM-DD | 代码 / 目标 / 文档 | {{改了什么}} | {{为什么改}} |
