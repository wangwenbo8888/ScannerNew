# 扫描查看器

> 实现位置：`modules/03_rendering/display/`（并入 `mod_rendering`）。本组件为渲染组件（OSG/CUDA-GL interop），非流水线算子，不适用算子规范 §2 三元组/Execute 契约。

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 显示-00 |
| 中文名称 | 扫描查看器 |
| 英文目录名 | scanner_viewer |
| 运行平台 | CPU + OpenGL（GPU） |
| 所属流程 | 显示渲染流程（封装入口） |
| 精度档次 | —（渲染，非测量） |

---

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 激光融合设备上下文 | `LaserCloudFuseDeviceContext`（透传给 `LaserCloudRenderer`） |
| **输入②** | 标记点显示数据 | `std::vector<MarkerCloudPoint>`（透传给 `MarkerCloudRenderer`） |
| **输出** | 渲染画面 | OSG 窗口（1280×720 默认） |

---

## C. 算法

**OSG 场景图结构**：

```
osgViewer::Viewer
  └── root (osg::Group)
      ├── laserGeode (osg::Geode)
      │   └── LaserCloudDrawable          ← 自定义 Drawable，CUDA-GL 零拷贝
      │       ├── GLSL Program (laser_vert + laser_frag)
      │       └── 3 VBO + 1 IBO (CUDA-GL interop)
      └── markerGeode (osg::Geode)
          └── osg::Geometry               ← 标准 OSG Geometry
              ├── GLSL Program (marker_vert + marker_frag)
              └── positions + uvs + normals + indices
```

**内联 GLSL 着色器（正反面区分）**：

| 着色器 | 功能 |
|--------|------|
| `kVertShader`（共用） | MVP 变换 + 传递 UV/Normal |
| `kLaserFragShader` | 正面蓝灰+法线光照 / 背面暗灰+弱光照 |
| `kMarkerFragShader` | 正面白圆+黑环 / 背面暗灰圆 |

---

## D. 依赖

**上下游算子**：

```
laser_cloud_fuse_cuda + laser_cloud_normal_cuda
    ↓ DeviceContext
[本算子] ScannerViewer
    ├── laserRenderer() → LaserCloudRenderer（零拷贝渲染）
    └── markerRenderer() → MarkerCloudRenderer（CPU 展点渲染）

point_reconstruct
    ↓ MarkerCloudPoint
[本算子] markerRenderer()
```

**子模块依赖**：

| 子模块 | 依赖 |
|--------|------|
| `laser_cloud_renderer` | `point_expand_kernel` + 融合算子 + OSG + CUDA-GL interop |
| `marker_cloud_renderer` | `point_expand_math.h` + `common/calib_logging.h` |
| 两者之间 | **无依赖**（完全解耦） |

**头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `laser_cloud_renderer.h` | 激光渲染器 |
| `marker_cloud_renderer.h` | 标记点渲染器 |
| `<osgViewer/Viewer>` | OSG 查看器 |
| `<osgGA/TrackballManipulator>` | 相机交互（鼠标旋转/缩放） |
| `<osg/Program>` / `<osg/Shader>` | GLSL 着色器程序 |

---

## D2. 衔接

**调用示例**：

```cpp
// 初始化
ScannerViewer viewer(1 << 20, 256);  // 100万激光点 + 256标记点
viewer.init(1280, 720);

// 扫描循环
while (!viewer.done()) {
    // ... 融合处理 ...
    auto ctx = fuse.GetDeviceContext();
    viewer.laserRenderer()->update(ctx);
    // 标记点的 R/T 变换+去重由 MarkerCloudFuseCPU 完成，渲染器只接收融合后的点
    viewer.markerRenderer()->update(fusedMarkers);
    viewer.markerRenderer()->flush();
    viewer.frame();
}
```

---

## E. 架构

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `ScannerViewer`（pImpl） |
| 初始化 | `init(width, height)` — 创建窗口 + 场景图 + shader + TrackballManipulator + 首帧 realize |
| 渲染 | `frame()` — 渲染一帧，返回 false 表示窗口关闭 |
| 激光渲染器 | `laserRenderer()` → `LaserCloudRenderer*` |
| 标记点渲染器 | `markerRenderer()` → `MarkerCloudRenderer*` |
| 状态查询 | `done()` — 窗口是否关闭 |
| 线程模型 | `SingleThreaded`（OSG + CUDA 同线程） |

**文件结构**：

```
display/
├── scanner_viewer.h               # 公开头文件
├── scanner_viewer.cpp             # 实现 + 内联 GLSL 着色器源码
├── display_benchmark.cpp          # 性能基准测试程序
└── CMakeLists.txt                 # 构建 scanner_display + display_benchmark
```

---

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenSceneGraph | 3.6.5 | osgViewer / osgGA / OpenThreads（osgDB/osgUtil 仅 CMake 传递链接，代码未直接调用） |
| CUDA Toolkit | ≥ 12.x | 展点 kernel + CUDA-GL interop |
| OpenGL | ≥ 3.3 | GLSL 330 core + `gl_FrontFacing` |
| 运行时 PATH | — | `C:\devlibs\osg-install\bin` |

---

## F. 参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `maxLaserPoints` | size_t | 1<<20 (100万) | 激光点云最大点数 |
| `maxMarkers` | size_t | 256 | 标记点最大累积数 |
| 窗口尺寸 | int×int | 1280×720 | OSG 窗口大小 |
| 背景色 | osg::Vec4 | (0.15, 0.15, 0.18, 1.0) | 深灰色 |
| 相机交互 | TrackballManipulator | — | 鼠标旋转/缩放/平移 |
| 光源方向 | vec3 | (0.4, 0.6, 0.8) | 法线光照方向（着色器内） |

---

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 渲染 FPS | 153 FPS（1M 点，RTX 5000） |
| 窗口模型 | `SingleThreaded`（非多线程渲染） |
| 首帧开销 | ~80ms（VBO 创建 + CUDA interop 注册 + 相机定位，一次性） |
| 深度测试 | 开启（`GL_DEPTH_TEST`） |
| 标记点可见性 | 空 Geometry（0 索引）天然不渲染，`flush()` 填充顶点后显示 |

---

## K. 质量

**端到端基准（100 万激光点 + 50 标记点/帧 × 100 帧）**：

| 步骤 | 耗时 |
|------|------|
| `update()`（激光，存储指针） | ~0 ms |
| `update()`（标记，存储融合点） | ~0 ms |
| `flush()`（标记，展点+上传） | 0.014 ms |
| `frame()`（GPU 展点 + 渲染） | 6.5 ms |
| **端到端** | **6.5 ms → 153 FPS** |

---

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
| **现有模块** | `display/scanner_viewer`（`ScannerViewer`） |
| **基准程序** | `display_benchmark.exe`（CMake BUILD_TESTS 时构建） |
| **子模块独立性** | `laser_cloud_renderer` 与 `marker_cloud_renderer` 完全解耦 |

---

> **文档结束**
