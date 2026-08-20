# AGENTS.md — AI 查阅入口（工程导航）

> AI 接手本工程先读本文件。真实实现以 `base/` `modules/` `app/` 与 `docs/` 为准。

### 优先查阅（AI 必读）
- **数据归属/管理** → 原则见 `数据管理原则.md`（L1–L4 分类 + 判定规则）；代码现状 / 修改目标 / 待增加见 `docs/模块功能/06-文件管理.md`
- **会话接力 / 当前进度** → `开发进度.md`（默认限读：快照+关键决策+近 10 条日志）
- **app 应用层** → `docs/应用层/README.md`（入口与启动 / AppContext 装配 / MainWindow UI / 构建与依赖）
- **base 共享内核** → `docs/共享内核/README.md`（types.h + EventBus 通俗说明）
- **禁区**：`factory_calib/` AI 只读保护区（见 `factory_calib/AGENTS.md`）

> **阶段**：framework/ 已退役删除（2026-08）——共享层职责落入 `base/` 与各业务模块；分层依赖单向：`base ← 06/07/08 ← 业务模块 ← app`（06/08 只链 base；07=mod_pipelinemgmt 链 base+mod_fileio+mod_operatorlib——五流水线对象消费容器与算子；08 的 HardwareMonitor 源码 include 06 的 DeviceStateCache，链接符号由 app 侧汇聚解析；10 链 base+spdlog，StateMachine 已自 07_session 迁入）；2026-08-20 模块 07 落地后 `scan_demo.exe` 全量构建 + ctest **68/68** 绿。`factory_calib/` 为独立厂家标定子工程（已交付，AI 只读保护区）。

---

## 顶层

```
JEAMMWARE260705/
├── base/           # 共享内核（types.h + EventBus；STATIC lib `base`；极薄，禁业务代码）
├── modules/        # 11 业务模块（mod_* 静态库 + 编入 scan_demo 的源文件）
├── app/            # 应用入口与装配层（scan_demo.exe；Qt5 UI + OSG）
├── cmake/          # 构建辅助（CompilerSettings / PatchVcxproj / PatchCudaVcxproj / Version.h.in）
├── factory_calib/  # 厂家标定独立子工程（产线交付物，AI 只读保护区，见下文专节）
├── docs/           # 流水线 / 模块功能 / 算子说明 / 应用层 / 共享内核 / 开发信息
├── build/          # Debug 构建产物（gitignore）
├── build-rel/      # Release 构建产物（gitignore）
└── CMakeLists.txt
```

### 历史注记（framework 退役，2026-08）
- `framework/` 整树已删：types.h+EventBus → `base/`；data 容器+Sink 契约、IWorkflow/Pipeline(Stage)、ParameterManager → `06_fileio`；CalibStore → `01_calibration`；hal 接口（IScannerCamera/IMCU）→ `08_devicemgmt`；service（StateMachine/SessionService/IState）→ `07_session`；FaultHandler → `10_observability`；工作流+WorkflowContext+ScannerWindow → `app/`。详见 `framework优化.md`。
- `sdk/`（接入端 B 契约桩）已删；旧根文档 `架构设计-简洁版.md` / `工程目录地图.md` 已删，导航职责归本文件。

---

## base/ — 共享内核（最底层公共零件箱）

极薄：只放公共类型 + 事件总线，**禁业务代码**；铁规"谁都依赖它，它不依赖任何人"（仅 C++ 标准库）。定名 `base/` 以避开 09 的 `core/` 撞名。

| 文件 | 职责 |
|---|---|
| `types.h` | 公共类型：`Result`（ok/fail/degraded/warning 工厂）、`QualityFlag`、`Pose`、`Event`/`EventType`、`DeviceState`、`ScanMode`、`UPtr`/`SPtr` |
| `EventBus.h/.cpp` | 线程安全事件总线（订阅/发布），编入 STATIC lib `base` |

通俗说明：`docs/共享内核/README.md`。

---

## modules/ — 11 业务模块

> ⚠ **编译归属注意**：`01` 的 UI 类、`02/04` 工作流、`03` OSGWidget、`06` file_io **物理在 modules/ 下但直接编入 app 的 scan_demo**（见 `app/CMakeLists.txt`）；`01(部分)/06/07/08/09/10` 另编为 `mod_*` **静态库**。`02/03/04/05/11` 各有同名 INTERFACE 占位库（mod_scanning / mod_rendering / mod_postprocessing / mod_editing / mod_deploy，无源码，仅保目标名）。

| 编号 | 模块 | 状态 | 内容 |
|---|---|---|---|
| 01 | `calibration` | 部分实现 | 库 `mod_calibration`=CalibStore；另有 CalibrationWorkflow / calib_workflow / CalibDialog / CalibDisplay / IntegrateTestDialog（编入 scan_demo；工作流帧处理已移交 07 A/B 对象，棋盘格链已退役） |
| 02 | `scanning` | 部分实现 | ScanWorkflow（编入 scan_demo；scanLoop 五 Stage 已移交 07 ScanPipeline，自身只留编排/记账） |
| 03 | `rendering` | 部分实现 | OSGWidget（编入 scan_demo）；`mod_rendering` 为 INTERFACE 占位（旧 display 渲染组件源码已退场，说明文档仍在 `docs/算子说明文档/display/`） |
| 04 | `postprocessing` | 部分实现 | PostProcessWorkflow（编入 scan_demo；旧七阶段 sleep 壳已移交 07 PostProcessPipeline 五阶段编排） |
| 05 | `editing` | 桩 | — |
| **06** | **`fileio`** | **✅ 承重** | 库 `mod_fileio`：运行时容器 FrameBuffer / PointCloudBuffer / DeviceStateCache / RingBuffer / SlotRing（原子槽位环）/ EnhancedFrame / CycleUnit / FrameEnricher（出口查表） + Sink 契约（IFrameSink/IPointCloudSink/IDeviceStateSink）+ IWorkflow/Pipeline(Stage) 工作流框架 + ParameterManager + file_io |
| 07 | `pipelinemgmt` | ✅ | 库 `mod_pipelinemgmt`：并行调度底座（sched/：CpuTopology/PCoreBroker/GpuSlotService/IFrameSource/FrameResultQueue/SchedulerRuntime）+ 五流水线对象（pipelines/：A 姿态判断 / B 标定计算 / C 扫描处理 / D 全局优化 / E 后处理）+ 装配公共件；命名空间 `Scanner::pipeline`。旧 `07_session` 已退役（StateMachine 迁 10、SessionService 随 D6/D7 定案撤销，会话记账归工作流自身） |
| 08 | `devicemgmt` | ✅ | 库 `mod_devicemgmt`：IScannerCamera / IMCU 接口 + CameraControl / MCUDriver / HardwareMonitor |
| **09** | **`operatorlib`** | **✅ 全部算子** | 单库 `mod_operatorlib`，命名空间 `calib::`（见下文专节）；GBA 含软先验扩展、marker_cloud_fuse 含 seed() |
| 10 | `observability` | 部分实现 | 库 `mod_observability`：StateMachine/IState（7 态表驱动 CAS，2026-08-20 自 07_session 迁入并重写）+ CommandGate 统一命令通道（双口）+ FaultHandler 故障档案表 + ObsLogger/jmw_logging + CrashHandler + PerfMonitor。**待建/待接**：08 检测源与性能指标源（归 08）、PerfMonitor 定时轮询、日志宏全模块推广 |
| 11 | `deploy` | 桩 | — |

---

## modules/09_operatorlib/ — 算子库（核心实现）

> 09 模块的核心实现层。单静态库 `mod_operatorlib`（core+calibration+scanning 合建）。算子规范见 `算子规范.md`，逐算子说明见 `docs/算子说明文档/`。

```
09_operatorlib/                         # 命名空间 calib::（3DSCANNER 机械迁入）
├── core/                               # ── 共享算子（标定/扫描双链复用）──
│   ├── common/                         #   共享类型头（非算子）
│   │   ├── calib_types.h               #     通用标定类型（StereoCalibration/TemperatureData…）
│   │   ├── calib_result_types.h        #     标定→扫描跨应用数据契约（StereoRectifyParams…）
│   │   ├── zitai_result_types.h        #     姿态结果类型
│   │   ├── quality_flag.h              #     质量标志位
│   │   ├── result.h                    #     统一 Result<T> 模板
│   │   ├── pipeline_types.h            #     流水线帧/上下文类型
│   │   ├── calib_warmup_config.h       #     Warmup 预热配置
│   │   ├── calib_logging.h             #     算子日志宏（spdlog 封装）
│   │   ├── json_utils.h                #     JSON 序列化工具
│   │   ├── scanner_api.h               #     对外 API 网关
│   │   └── version.h                   #     版本号
│   ├── vision/                         #   底层视觉核 [CUDA]（双链共用前端）
│   │   ├── mask_extract/               #     激光掩膜区域提取
│   │   ├── frame_filter/               #     帧类型过滤（排除激光线帧，姿态判断链 2-1b）
│   │   └── ccl/                        #     连通域分析（region_analyze）
│   ├── marker/                         #   标记点链 [CPU]
│   │   ├── image_split/                #     标记点图像分割
│   │   ├── zernike_edge/               #     Zernike 椭圆边缘亚像素提取
│   │   ├── image_merge/                #     标记点图像合并
│   │   ├── undistort_cpu/              #     双目去畸变矫正
│   │   ├── ellipse_fit/                #     椭圆拟合中心提取
│   │   ├── marker_match/               #     标记点双目匹配
│   │   ├── epipolar_intersect/         #     椭圆边界极线交点
│   │   ├── edge_match/                 #     椭圆边缘点双目匹配
│   │   ├── point_reconstruct/          #     标记点三维重建
│   │   ├── frame_fuse/                 #     单帧配准（兜底）
│   │   ├── optical_flow_fuse/          #     标记点光流配准（默认）
│   │   └── marker_cloud_fuse_cpu/      #     标记点点云体素哈希融合
│   ├── laser/                          #   激光线链 [CUDA]
│   │   ├── steger/                     #     Steger 激光中心亚像素提取
│   │   ├── undistort_cuda/             #     去畸变矫正（CUDA）
│   │   ├── epipolar_interp/            #     极线插值
│   │   └── laser_reconstruct/          #     激光线三维重建
│   └── scheduler/                      #   调度层状态
│       └── prev_frame_state.h          #     AtomicFrameState（跨帧状态）
├── calibration/                        # ── 标定独有算子 ──
│   ├── camera/                         #   相机标定链 [CPU]
│   │   ├── inverse_distort/            #     逆畸变变换
│   │   ├── intrinsic_calib/            #     双目内参标定
│   │   ├── extrinsic_calib/            #     双目外参标定
│   │   ├── stereo_rectify/             #     立体矫正
│   │   └── stereo_rectify_temp_table/  #     温度补偿立体矫正参数表
│   ├── laser_calib/                    #   虚拟相机标定 [CUDA/混合]
│   │   ├── laser_label/                #     激光线编号
│   │   ├── laser_match/                #     激光线匹配（标定版）
│   │   ├── endpoint_extract/           #     激光线 3D 端点提取
│   │   ├── virtual_camera_pose/        #     虚拟相机光心求解
│   │   ├── pose_optimize/              #     虚拟相机外参优化
│   │   ├── projector_joint_calib/     #     投影机光心与发射曲线联合标定（替代端点链路，独立）
│   │   ├── plane_map/                  #     激光平面映射表
│   │   └── plane_map_temp_table/       #     温度补偿平面映射查找表
│   ├── posture/                        #   姿态估计 [CPU]
│   │   └── pose_estimate/              #     设备姿态估计
│   └── temp/                           #   温度补偿 [CPU]
│       ├── intrinsic_compensate/       #     内参温度补偿
│       ├── extrinsic_compensate/       #     外参温度补偿
│       └── laser_extrinsic_compensate/ #     激光外参温度补偿
└── scanning/                           # ── 扫描独有算子 ──
    ├── preprocess/                     #   联合掩膜分离 [CUDA]
    │   └── mask_separation/            #     激光标记点掩膜分离
    ├── laser/                          #   激光线匹配-扫描 [CUDA]
    │   └── laser_match_scan/           #     激光线匹配扫描（查温度映射表）
    ├── fusion/                         #   点云融合 [CPU+CUDA]
    │   ├── laser_cloud_fuse/           #     激光点云体素哈希融合（CPU）
    │   ├── laser_cloud_fuse_cuda/      #     激光点云体素哈希融合（CUDA）
    │   ├── laser_cloud_normal/         #     激光点云法线估计（CPU）
    │   └── laser_cloud_normal_cuda/    #     激光点云法线估计（CUDA）
    └── global_optim/                   #   全局优化（Ceres）※ 非算子三元组
        ├── global_ba_cpu.{h,cpp}       #     全局 Bundle Adjustment（入库，BUILD_GLOBAL_OPTIM）
        ├── ba_residuals.h              #     BA 残差块
        ├── pose_graph_residuals.h      #     位姿图残差块
        ├── gba_dataset_runner.cpp      #     GBA 离线工具 exe（不入库）
        └── tests/                      #     GBA 测试
```

**流水线（处理顺序）**：
- 标记点链（core/marker）：image_split→zernike_edge→image_merge→undistort_cpu→ellipse_fit→marker_match→epipolar_intersect→edge_match→point_reconstruct→[frame_fuse | optical_flow_fuse]→marker_cloud_fuse_cpu
- 激光线链（core/laser）：steger→undistort_cuda→epipolar_interp→laser_reconstruct
- 两条链前端共用 core/vision（mask_extract、ccl）

每个算子目录约定：`<name>_cuda.h` / `.cpp` / `_pimpl.h` / `_impl.cu` / `tests/test_*.cpp`。算子契约：三元组（Params/Result/Operator）+ `Execute()`/`Destroy()`/`Warmup()` + pImpl + `Stream&` 流亲和。

---

## app/ — 应用入口与装配层（组合根）

产物 `scan_demo.exe`（Qt5 Widgets + Svg/SerialPort + OSG）。app 不实现业务逻辑，只做装配与依赖注入：`AppContext` 拥有全部运行时组件，经 `WorkflowContext` 窄接口注入；分层装配 + 逆序析构。

```
app/
├── main.cpp                  # 入口（Qt/OSG/spdlog 初始化 → 装配 AppContext → 全屏主窗口）
├── AppContext.h/.cpp         # 装配根
├── MainWindow.h/.cpp         # 主窗口 UI（无边框 "LeadScan K2"）
├── WorkflowContext.h/.cpp    # 工作流注入窄接口
├── ScannerWindow.h/.cpp/.ui  # 扫描窗口 UI
├── stubs/                    # 外部依赖桩头（LEADSCANSeries.h 等 5 个，人工提供；scan_demo 条件构建守卫）
├── copy_dlls.bat             # POST_BUILD 拷贝 Qt/OSG/OpenCV 运行时 DLL
├── resources.qrc + resources/icons/  # Qt 资源（三态 SVG 图标）
└── dark.qss                  # 暗色主题（未登记进 qrc，暂不生效）
```

详见 `docs/应用层/`（README + 01-入口与启动 / 02-AppContext装配 / 03-MainWindow-UI / 04-构建与依赖）。

---

## factory_calib/ — 厂家标定独立子工程

> ⚠ **AI 只读保护区**（见 `factory_calib/AGENTS.md`）：除非人工明确指令，AI 不得读取/修改/构建/测试本目录。下述仅作目录索引。

两个独立可执行（自包含 CMake 工程，依赖与主工程一致：OpenCV 4.13 / CUDA 12.6 / Eigen 3.4.1）：
- `module1_camera/camera_calib.exe` — 相机内/外参 + 立体矫正 + 温度补偿表（CPU）
- `module2_laser/laser_calib.exe` — 激光虚拟相机标定 + 温度补偿表（CUDA，CLI 仅 Release）

```
factory_calib/
├── CMakeLists.txt        # 自包含构建（FC_BUILD_MODULE1/MODULE2 开关）
├── module1_camera/       # 相机标定 exe（CPU，无 CUDA）
├── module2_laser/        # 激光标定 exe（CUDA，CLI 仅 Release）
├── cmake/                # 构建辅助（fc_deploy_crt 等）
├── data_in/              # 输入样本（camera/ 棋盘格 · laser/ pose_xx/）
├── data_out/             # 输出（camera_calib.json · laser_calib.json）
├── build_fc1_rel/        # 模块1 Release 构建产物（gitignore）
├── build_fc2_rel/        # 模块2 Release 构建产物（gitignore）
├── AGENTS.md             # AI 访问受限声明
├── README.md             # 构建命令 / 输入输出约定 / 已知限制 / 测试覆盖
└── .gitignore
```

设计与实现见 `factory_calib/README.md`（AI 只读）。测试 26/26 绿（模块1 8/8 + 模块2 18/18，Release）。

---

## docs/

```
docs/
├── 流水线/            # 三条流水线独立完整描述 + README 索引
│                        #   客户端标定 / 客户端扫描 / 03-厂家标定(factory_calib)
│                        #   另有并行调度方案 2 份（扫描并行调度 / 姿态判断并行调度）
├── 模块功能/          # 11 业务模块功能文档（01-标定工作流 … 11-安装部署），索引见根 模块功能目录.md
├── 算子说明文档/       # 46 份算子/渲染组件说明（A–K 模板），自带索引 算子目录.md，
│                        #   分 core/(19) calibration/(17) scanning/(7) display/(3)
├── 应用层/            # app/ 核心要点 5 份（README / 01-入口与启动 / 02-AppContext装配 /
│                        #   03-MainWindow-UI / 04-构建与依赖）
├── 共享内核/          # base/ 通俗说明（README）
├── plans/             # 实施计划存档（当前已清空，历史方案文档已删）
└── 开发需要的信息/     # 大恒 Galaxy SDK 文档（C++接口为核心）+ 相机(VE2S-301-125U3MC-S)数据手册
                         #   + 下位机和按键/（协议/按键资料）
```

---

## 构建速查

**配置**（根 `CMakeLists.txt`）：C++17 / CUDA 17 / `CMAKE_CUDA_ARCHITECTURES=75;86;87` / nvcc pin v12.6 / `/MD(D)` CRT；`BUILD_UI=OFF`（暂缓）、`BUILD_GLOBAL_OPTIM=ON`。构建辅助脚本在 `cmake/`。

**依赖**：OpenCV 4.13（Release `C:/opencv-cuda-4.13.0` / Debug 自建 `C:/opencv-cuda-4.13.0-debug`）、Eigen 3.4.1（`C:/devlibs/eigen-3.4.1-install`）、**Qt 5.15.2**（`C:/devlibs/Qt-5.15.2-msvc2019_64`；scan_demo 需 Core/Gui/Widgets/OpenGL/Svg/SerialPort）、**OSG**（`C:/devlibs/osg-install`）；Ceres 2.2 / spdlog 1.15 / nlohmann_json / gtest 经 FetchContent（github 被封时 `-DJMW_GH_MIRROR=https://ghfast.top/`）。路径详见 `环境配置汇总.md`。

**scan_demo 条件构建**：仅当 Qt5 Svg/SerialPort 齐备 **且** `app/stubs/LEADSCANSeries.h` 存在时构建，否则跳过以保"库+测试"构建绿。

> ⚠ 现有 `build/` 是 **Debug-only**（`CMAKE_CONFIGURATION_TYPES=Debug` + Debug OpenCV）。直接在 `build/` 跑 Release 会报 `MSB8013`。Release 请用独立目录 `build-rel`。

**Release**（独立目录，实测 43/43 绿）：
```powershell
cmake -S . -B build-rel -G "Visual Studio 17 2022" -A x64 -DCMAKE_CONFIGURATION_TYPES=Release
cmake --build build-rel --config Release
ctest --test-dir build-rel -C Release --output-on-failure
```

**Debug**（现有 build/，实测 43/43 绿）：
```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DOpenCV_DIR=C:/opencv-cuda-4.13.0-debug/x64/vc17/lib `
  -DCMAKE_CONFIGURATION_TYPES=Debug `
  -DJMW_GH_MIRROR=https://ghfast.top/
cmake --build build --config Debug
$env:PATH = 'C:\opencv-cuda-4.13.0-debug\x64\vc17\bin;' + $env:PATH
ctest --test-dir build -C Debug --output-on-failure
```

---

## 关键根文件

| 文件 | 用途 |
|---|---|
| `CMakeLists.txt` | 工程构建配置（C++/CUDA 标准、依赖、FetchContent） |
| `框架整体.md` | 框架整体设计（01.4 框架图：5 层主架构 + 部署期 wrapper + 横切） |
| `framework优化.md` | framework/ 退役决策与归属迁移记录（2026-08） |
| `模块功能目录.md` | `docs/模块功能/` 下 11 模块文档的目录与简介索引（仅导航，非代码改动基准） |
| `数据管理原则.md` | 数据归属原则：L1–L4 分类 + 判定规则（现状/修改目标/待增加见 `docs/模块功能/06-文件管理.md`） |
| `算子规范.md` | 算子契约 + 工程集成规范（v2.0，§0–§7） |
| `算子目录.md` | 算子说明文档目录索引（导航至 `docs/算子说明文档/` 下 46 份说明） |
| `算子提炼模板.md` | 算子开发参考卡模板（v1.0，12 段落） |
| `开发进度.md` | AI 会话接力状态（当前快照 + 会话日志 + 关键决策，含限读规则） |
| `环境配置汇总.md` | 本机开发环境（VS/CUDA/OpenCV/Qt/OSG/三方库路径） |
| `AGENTS.md` | 本文件（AI 查阅入口 / 工程目录地图） |
