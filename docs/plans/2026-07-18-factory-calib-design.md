# 厂家标定工程 — 设计文档

> **日期**: 2026-07-18
> **状态**: 已通过逐节评审，待生成实现计划
> **位置**: 工程根目录下新建 `factory_calib/`
> **来源需求**: 用户提出 —— 两个独立模块（相机标定 / 激光标定），尽量复用主工程算子，整个文件夹自包含可独立运行

---

## 0. 背景与目标

主工程（JEAMMWARE260705）的标定流水线面向产品软件运行期，深度耦合 framework 层与扫描姿态判断链。厂家产线需要一个**自包含、离线、可直接拷贝部署**的标定工具：

- **模块1（相机标定）**：相机内参（改用棋盘格）/ 外参 / 立体矫正 / 3 张温度补偿表
- **模块2（激光标定）**：激光虚拟相机标定 / 2 张温度补偿表
- 两模块各自独立 exe，文件交接
- 整个 `factory_calib/` 文件夹**不引用主工程任何源码**，所需要算子原样拷贝进来
- 尽量复用主工程既有算子，只新增"棋盘格角点提取+编号"这一个自实现函数

---

## 1. 顶层结构

```
factory_calib/                       # 工程根目录下新建
├── README.md                        两模块构建/运行说明 + 目录约定
├── CMakeLists.txt                   顶层，选项 BUILD_MODULE1 / BUILD_MODULE2
├── module1_camera/                  ★相机标定（纯 CPU，不依赖 CUDA）
│   ├── CMakeLists.txt               独立可单独配置，产出 camera_calib.exe
│   ├── operators/                   从主工程拷贝的算子（自包含副本）
│   │   ├── common/                  core/common 公共头副本
│   │   ├── intrinsic_calib/         复用（求解）
│   │   ├── extrinsic_calib/         复用
│   │   ├── stereo_rectify/          复用
│   │   ├── intrinsic_compensate/    复用（内参温度表）
│   │   ├── extrinsic_compensate/    复用（外参温度表）
│   │   └── stereo_rectify_temp_table/  复用（立体矫正温度表）
│   ├── chessboard_corner.{h,cpp}    ★自实现：提取+编号 单 CPU 函数
│   ├── calib_io.{h,cpp}             读图/读温度/写参数文件
│   ├── camera_calib_cli.cpp         main 入口
│   └── tests/
├── module2_laser/                   ★激光标定（CUDA）
│   ├── CMakeLists.txt               独立可单独配置，产出 laser_calib.exe
│   ├── operators/                   从主工程拷贝的算子（自包含副本）
│   │   ├── common/                  core/common 公共头副本
│   │   ├── vision/                  mask_extract, ccl（region_analyze）
│   │   ├── laser/                   steger, undistort_cuda, epipolar_interp, laser_reconstruct
│   │   ├── laser_calib/             laser_label, laser_match, endpoint_extract,
│   │   │                            virtual_camera_pose, pose_optimize, plane_map,
│   │   │                            plane_map_temp_table
│   │   ├── extrinsic_compensate/    传递依赖（被 laser_extrinsic_compensate 包含）
│   │   └── laser_extrinsic_compensate/  复用（激光外参温度表）
│   ├── calib_io.{h,cpp}             读图/读模块1输出/写参数文件
│   ├── laser_calib_cli.cpp          main 入口
│   └── tests/
├── data_in/                         示例/约定输入布局（README 描述）
└── data_out/                        默认输出位置
```

**结构要点**：
- 两个模块各自独立 CMake、独立 exe，**无代码共享**（仅 `core/common` 头各拷一份）
- 模块1 不拷任何 CUDA 代码 → 可在无 CUDA 环境构建运行
- 不拷贝阶段2 标记点链（14 算子）—— 离线已预选姿态

---

## 2. 模块1 数据流（相机标定，CPU）

```
磁盘: left/{001..N}.png + right/{001..N}.png + temps.txt(参考温度)
  │
  ▼  对每帧 (L,R) 调一次
┌─────────────────────────────────────────────────────┐
│ ★ extractChessboardCorners(gray, pattern, out)        │  ← 自实现单 CPU 函数
│   内部: findChessboardCornersSB + cornerSubPix 亚像素  │
│   输出: vector<Point2f> 行主序、L/R 编号一致            │
│   + L/R 一致性校验(必要翻转，保证同物理点对齐)          │
└───────────────────┬─────────────────────────────────┘
                    │ left_points[N], right_points[N]
        ┌───────────┴───────────┐
        ▼                       ▼
┌──────────────────────┐  ┌───────────────────────────────┐
│ intrinsic_calib (3-2) │  │ (内参 L/R 喂给下游)             │
│ 复用，求解内参+畸变 L/R │  │                               │
│ → IntrinsicCalibResult│  │                               │
└──────────┬───────────┘  │                               │
           │              ▼
           │      ┌──────────────────────┐  ┌──────────────────────┐
           │      │ extrinsic_calib (3-3) │  │ stereo_rectify (3-4)  │
           │      │ 复用 → R,T,E,F         │  │ 复用 → R1,R2,P1,P2,Q  │
           │      └──────────┬───────────┘  └──────────┬───────────┘
           │                 │                         │
           ▼                 ▼                         ▼
┌──────────────────┐ ┌────────────────────┐ ┌────────────────────────────┐
│intrinsic_compens│ │extrinsic_compensate │ │stereo_rectify_temp_table   │
│ate (5-1) 复用    │ │(5-2) 复用            │ │(3-5) 复用                   │
│→ 内参温度表 L/R  │ │→ 外参温度表          │ │→ 立体矫正温度表             │
└──────────────────┘ └────────────────────┘ └────────────────────────────┘
           │                 │                         │
           └────────┬────────┴─────────────────────────┘
                    ▼
            写 camera_calib.json  → 供模块2 消费
```

**要点**：
- **跳过 `inverse_distort`(3-1)**：棋盘角点直接在原始畸变图像空间提取，无需像标记点链那样在矫正空间往返
- **内参求解复用 `intrinsic_calib` 算子**：它本来就吃 `vector<vector<Point2f>>` + 棋盘格参数（`chessboard_width/height/square_size_mm`）+ 温度系数，**不需改它的代码**，只换"点的来源"
- **`extractChessboardCorners` 单函数职责**：输入灰度图+棋盘规格 → 输出行主序角点 + `found` 标志。L/R 一致性由 SB 标志位 + 必要翻转保证（同一函数，每帧调用一次）
- **温度输入**：`temps.txt` 给参考温度（标定时的环境/相机温度），喂给算子 `referenceTemp` 字段
- 模块1 不需要 GPU，全程 CPU

---

## 3. 模块2 数据流（激光标定，CUDA）

模块2 = 主工程阶段4（激光虚拟相机标定）**原样搬运**，只换数据来源：磁盘读图 + 读模块1 输出文件，不做算法改动。

```
磁盘: pose_00../pose_24/  每姿态下 left/right 激光图(多管) + temp
       + camera_calib.json (← 模块1 输出: 内参/畸变/R/T/R1/R2/P1/P2/Q/refTemp/CTE)
  │
  ▼  逐姿态、逐激光管帧
┌───────────────────────────────────────────────────────┐
│ 4-1  mask_extract (CUDA)         ── 激光掩膜提取        │
│ 4-2  ccl region_analyze (CUDA)   ── 连通域分析          │
│ 4-3  laser_label (CUDA)          ── 激光线按Y编号        │
│ 4-4  steger (CUDA, ByLabel)      ── Steger 中心亚像素    │
│ 4-5  undistort_cuda (CUDA)       ── 用模块1 内参去畸变   │
│ 4-6  epipolar_interp (CUDA, lineIdCheck=true)          │
│ 4-7  laser_match (CUDA)          ── L/R 同线匹配         │
│ 4-8  laser_reconstruct (CUDA)    ── 三维重建(吃模块1 Q)  │
└───────────────────┬───────────────────────────────────┘
                    │ 累积多姿态 d_points3d + line_ids
        ┌───────────┴───────────┐
        ▼                       ▼
┌──────────────────────┐  ┌───────────────────────────────┐
│ 4-9  endpoint_extract│  │ 4-11 pose_optimize (CUDA)      │
│      (CUDA)          │  │   吃 4-8 + 4-10                │
└──────────┬───────────┘  └──────────┬────────────────────┘
           ▼                         │
┌──────────────────────┐             │
│ 4-10 virtual_camera_ │◄────────────┘
│      pose (CUDA)     │  吃模块1 stereoK/stereoR
│ → virtualK/R/T       │────────────────────────────┐
└──────────────────────┘                            │
           │ (优化后 virtualK/R/T ← 4-11)            │
           ▼                                        ▼
┌──────────────────────┐  ┌──────────────────────────────────┐
│ 4-12 plane_map(CUDA) │  │ 5-3 laser_extrinsic_compensate(CPU)│
│   吃模块1 StereoCalib │  │   吃 4-11 virtual→L/R 外参         │
│   → 激光面映射表       │  │   → 激光外参温度表                  │
└──────────┬───────────┘  └───────────────────────────────────┘
           ▼
┌────────────────────────────────┐
│ 4-13 plane_map_temp_table(CUDA)│
│   吃模块1 立体标定 + virtual    │
│   → 温度补偿映射表               │
└──────────┬─────────────────────┘
           ▼
   写 laser_calib.json
```

**要点**：
- **算子零改动**：模块2 只是"驱动层"，所有 CUDA 算子 `.cu`/`.cpp`/pimpl 原样拷贝；参数从模块1 输出文件加载填入
- **循环结构**：4-1~4-8 按姿态×激光管帧循环（累积 3D 点）；4-9~4-13 在集齐后一次性执行
- **模块1 → 模块2 数据契约**：`camera_calib.json` 提供 `cameraMatrixL/R`、`distCoeffsL/R`、`R`、`T`、`R1`、`R2`、`P1`、`P2`、`Q`、`referenceTemp`、`cte`、`imageSize`、`tempRangeMin/Max/Step`（保证温度表范围一致）
- 模块2 全程需 CUDA + OpenCV-CUDA

---

## 4. 算子拷贝清单

### 4.1 模块1（CPU，共 6 算子 + 公共头）

| # | 算子目录（来自 modules/09_operatorlib/） | 文件 |
|---|---|---|
| 1 | `calibration/camera/intrinsic_calib/` | `intrinsic_calib_cpu.{h,cpp}` |
| 2 | `calibration/camera/extrinsic_calib/` | `extrinsic_calib_cpu.{h,cpp}` |
| 3 | `calibration/camera/stereo_rectify/` | `stereo_rectify_cpu.{h,cpp}` |
| 4 | `calibration/camera/stereo_rectify_temp_table/` | `stereo_rectify_temp_table_cpu.{h,cpp}` |
| 5 | `calibration/temp/intrinsic_compensate/` | `intrinsic_compensate_cpu.{h,cpp}` |
| 6 | `calibration/temp/extrinsic_compensate/` | `extrinsic_compensate_cpu.{h,cpp}` |

### 4.2 模块2（CUDA，共 15 算子 + 公共头）

| # | 算子目录 | 文件（含 `_pimpl.h` + `_impl.cu`） |
|---|---|---|
| 1 | `core/vision/mask_extract/` | `mask_extract_cuda.{h,cpp}` + `_pimpl.h` + `_impl.cu` |
| 2 | `core/vision/ccl/` | `region_analyze_cuda.{h,cpp}` + `_pimpl.h` + `_impl.cu` |
| 3 | `core/laser/steger/` | `steger_extract_cuda.*` + pimpl + cu |
| 4 | `core/laser/undistort_cuda/` | `undistort_points_cuda.*` + pimpl + cu |
| 5 | `core/laser/epipolar_interp/` | `epipolar_interp_cuda.*` + pimpl + cu |
| 6 | `core/laser/laser_reconstruct/` | `laser_reconstruct_cuda.*` + pimpl + cu |
| 7 | `calibration/laser_calib/laser_label/` | `laser_label_cuda.*` + pimpl + cu |
| 8 | `calibration/laser_calib/laser_match/` | `laser_match_cuda.*` + pimpl + cu |
| 9 | `calibration/laser_calib/endpoint_extract/` | `endpoint_extract_cuda.*` + pimpl + cu |
| 10 | `calibration/laser_calib/virtual_camera_pose/` | `virtual_camera_pose_cuda.*` + pimpl + cu |
| 11 | `calibration/laser_calib/pose_optimize/` | `pose_optimize_cuda.*` + pimpl + cu |
| 12 | `calibration/laser_calib/plane_map/` | `plane_map_cuda.*` + pimpl + cu + `virtual_pixel_gen.h` |
| 13 | `calibration/laser_calib/plane_map_temp_table/` | `plane_map_temp_table.{h,cpp}`（用 GpuMat） |
| 14 | `calibration/temp/extrinsic_compensate/` | `extrinsic_compensate_cpu.{h,cpp}`（传递依赖） |
| 15 | `calibration/temp/laser_extrinsic_compensate/` | `laser_extrinsic_compensate_cpu.{h,cpp}` |

### 4.3 公共头（`core/common/`，两模块各拷一份）

- **确定需要**：`calib_types.h`、`calib_result_types.h`、`result.h`、`quality_flag.h`、`json_utils.h`、`scanner_api.h`、`version.h`、`calib_logging.h`
- **按需**（实现期按 include 裁剪）：`pipeline_types.h`、`calib_warmup_config.h`、`zitai_result_types.h`

### 4.4 拷贝规则

- **原样拷贝**（不改算子代码），只调整 `#include "common/xxx.h"` 的相对路径前缀
- 不拷贝：阶段2 标记点链（14 算子）、scanning 算子、global_optim、rendering、inverse_distort
- 每算子自带 `tests/` 一并拷贝，作为迁移后回归（见 §7）

---

## 5. I/O 数据契约

### 5.1 模块1 输入（`data_in/camera/`）

```
camera/
├── config.json         棋盘格+算子参数
├── temps.txt           参考温度 (单行: "ref_temp 22.5")
├── left/  {001..N}.png     棋盘格图（CV_8UC1 或转灰度）
└── right/ {001..N}.png
```

`config.json` 关键字段：

```json
{
  "chessboard": {"width": 11, "height": 8, "square_size_mm": 2.0},
  "image_size": [2048, 1536],
  "intrinsic_flags": 0,
  "use_calibrateCameraRO": true,
  "reproj_error_threshold": 0.012,
  "temperature": {
    "cte": 23.6e-6,
    "referenceTemp": 22.5,
    "tempRangeMin": -10.0,
    "tempRangeMax": 10.0,
    "tempStep": 0.2
  },
  "rectify": {"alpha": 0.0, "flags": 1}
}
```

### 5.2 模块1 输出（`data_out/camera_calib.json`）

单文件，复用算子既有 `toJson()` 序列化：

- `cameraMatrixL/R`, `distCoeffsL/R`
- `R`, `T`, `E`, `F`
- `R1`, `R2`, `P1`, `P2`, `Q`, `validRoiL/R`
- `referenceTemp`, `cte`, `imageSize`, `tempRangeMin/Max/Step`（**回写，供模块2 对齐**）
- `intrinsicTempTableL/R`、`extrinsicTempTable`、`stereoRectifyTempTable`
- 精度指标（reproj/epipolar 误差）+ `qualityFlag`

### 5.3 模块2 输入（`data_in/laser/`）

```
laser/
├── config.json              激光链参数 (gridStep/depthMin/Max/epipolarStep/lineIds…)
├── camera_calib.json        ← 拷贝/软链自 data_out/camera_calib.json
├── pose_00/
│   ├── L_tube0.png  R_tube0.png
│   ├── L_tube1.png  R_tube1.png
│   └── temp.txt
├── pose_01/  ...
└── pose_24/
```

### 5.4 模块2 输出（`data_out/laser_calib.json`）

- `virtualK/R/T`（优化后）、`lineIds`
- `planeMap`（uL/vL/uR/lid + right_u）
- `laserExtrinsicTempTable`（virtual→L/R 每温度）
- `planeMapTempTable`（每温度映射）
- `qualityFlag`

### 5.5 序列化与一致性

- 统一 **nlohmann/json**（主工程既有依赖，算子已自带 `toJson/fromJson`）。备选 OpenCV `FileStorage` YAML。
- 模块2 启动时校验 `camera_calib.json` 的 `tempRangeMin/Max/Step`、`cte`、`imageSize` 与自身 `config.json` 一致，不一致则报错退出。

---

## 6. 构建、依赖、自包含性

### 6.1 顶层 `factory_calib/CMakeLists.txt`

- C++17 / MSVC v144
- 选项 `BUILD_MODULE1=ON`、`BUILD_MODULE2=ON`（默认都 ON；关 MODULE2 可在无 CUDA 环境只构建模块1）
- `add_subdirectory(module1_camera)` / `add_subdirectory(module2_laser)`
- `enable_language(CUDA)` 仅在 `BUILD_MODULE2=ON` 时触发；`CMAKE_CUDA_ARCHITECTURES=75;86;87`（同主工程）

### 6.2 各子模块 CMakeLists（各自独立可单独配置）

```powershell
cmake -S factory_calib/module1_camera -B build_m1   # 无需 CUDA
cmake -S factory_calib               -B build_all   # 全量
```

### 6.3 外部依赖（与主工程同版本，不引用主工程源码）

| 依赖 | 版本 | 方式 | 模块1 | 模块2 |
|---|---|---|:---:|:---:|
| OpenCV | 4.13（CPU） | `find_package(OpenCV)` (`C:/opencv-cuda-4.13.0[-debug]`) | ✓ | ✓ |
| OpenCV-CUDA | 同上（`cv::cuda::*`） | 同上（仅 Release 版带 CUDA） | — | ✓ |
| CUDA Toolkit | 12.6 / nvcc pin | `enable_language(CUDA)` | — | ✓ |
| Eigen | 3.4.1 | `find_package(Eigen3)` | ✓ | ✓ |
| nlohmann_json | — | `FetchContent`（带 `JMW_GH_MIRROR` 兼容） | ✓ | ✓ |
| GoogleTest | — | `FetchContent`（仅 `BUILD_TESTING=ON`） | ✓ | ✓ |

### 6.4 自包含性边界

- **不引用** `../../framework/`、`../../modules/`、`../../sdk/`、`../../app/`（完全不链接主工程）
- 算子副本的 `#include "common/xxx.h"` 改为指向各自 `operators/common/`
- 顶层 README 给出与主工程一致的 OpenCV/CUDA 路径与镜像开关，保证本机构建复现

### 6.5 产物

- `module1_camera/camera_calib.exe`（CPU）
- `module2_laser/laser_calib.exe`（CUDA）

---

## 7. 测试与验证

### 7.1 测试金字塔

| 层级 | 内容 | 工具 | CUDA |
|---|---|---|:---:|
| L1 算子回归 | 拷贝主工程每算子自带 `tests/test_*.cpp`（21 个）原样跑通 | CTest + gtest | 部分 |
| L2 单元 | `test_chessboard_corner`：OpenCV 合成棋盘图（已知角点）→ 验证提取数量/编号一致/亚像素精度 | gtest | 否 |
| L3 端到端 | `test_camera_calib_e2e`：合成已知内参的棋盘图组 → 跑模块1 → 验证恢复内参误差 < tol、reproj < 0.012px、3 张温度表非空 | gtest | 否 |
| L3 端到端 | `test_laser_calib_e2e`：合成激光线图（已知 virtual pose）或主工程 fixture → 跑模块2 → 验证映射表/温度表生成、virtualK/R/T 收敛 | gtest | 是 |
| L4 交接 | `test_handoff`：模块1 输出 → 模块2 加载，校验 schema + 温度范围一致性 | gtest | 否 |

### 7.2 合成数据策略

- **棋盘**：`cv::drawChessboardCorners` 配合 `cv::warpPerspective` + 畸变模型，生成已知 ground-truth 的图组（模块1 自给自足，无需真实采样）
- **激光**：优先复用主工程算子 `tests/` 内既有 fixture；若不足，合成"已知平面→投影激光线"图像组

### 7.3 成功判据

- 构建：Debug + Release 双配置均通过（独立 build 目录，同主工程规范）
- CTest：所有用例绿
- 模块1：reproj ≤ 0.012 px、3 张温度表填充完整、QualityFlag=Normal
- 模块2：plane_map + 2 张温度表生成、virtual pose 优化收敛

### 7.4 错误处理

沿用算子既有 `QualityFlag` + `success/message` 模型；CLI 驱动层遇到 `success=false` 以非零码退出并在 stderr 打印 `message`，不写半成品输出文件。

### 7.5 已识别风险

1. **`intrinsic_calib` 算子是否真兼容棋盘格点** —— 头文件签名通用，但实现期需验证其 object point 生成/温度补偿与棋盘格语义一致（若有 marker 假设则需小改）
2. **L/R 角点编号一致性** —— `findChessboardCornersSB` 在不同视角/遮挡下可能给不同原点；需在 `extractChessboardCorners` 内加方向归一化 + 极线残差校验，失败则丢帧
3. **模块2 E2E 合成数据复杂** —— 若主工程无现成 fixture，可能降级为"需真实样本数据，手动验证"

---

## 8. 决策记录

| 维度 | 决定 | 备选 | 理由 |
|---|---|---|---|
| 角点提取 | OpenCV `findChessboardCornersSB` 封装单函数 | 纯自实现 / 直接裸调 | 用户指定 |
| 输入来源 | 纯离线读盘 | 在线相机 SDK / 混合 | "独立运行" + 不依赖特定相机 SDK |
| 模块形态 | 两个独立 exe + 文件交接 | 一个 exe 两子命令 / 一键跑完 | "两个独立模块" |
| 数据布局 | 按帧/姿态平铺目录 | 配置文件显式列举 / 内置姿态判断 | 离线预选姿态，省 14 算子 |
| 代码组织 | 方案 B（两棵独立子树） | 方案 A（共享算子库） | 模块1 可无 CUDA 独立构建 |
| 内参求解 | 复用 `intrinsic_calib` 算子 | 自写 cv::calibrateCameraRO 包装 | "尽量复用主工程算子" |

---

## 9. 后续

下一步：调用 `writing-plans` 技能生成分阶段实现计划（拷贝迁移 → chessboard_corner → 模块1 驱动 → 模块1 测试 → 模块2 驱动 → 模块2 测试 → README/收尾）。
