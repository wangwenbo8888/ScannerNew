# 激光点云渲染器

> 实现位置：`modules/03_rendering/display/`（并入 `mod_rendering`）。本组件为渲染组件（OSG/CUDA-GL interop），非流水线算子，不适用算子规范 §2 三元组/Execute 契约。

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 显示-01 |
| 中文名称 | 激光点云渲染器 |
| 英文目录名 | laser_cloud_renderer |
| 运行平台 | CUDA + OpenGL（GPU） |
| 所属流程 | 显示渲染流程（独立流程，消费融合输出） |
| 精度档次 | —（渲染，非测量） |

---

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 融合设备上下文（位置 + 法线 + 体素边长 + 点数） | `LaserCloudFuseDeviceContext`（POD，GPU 指针） |
| **输出** | OSG Drawable 节点（挂到场景图即可渲染） | `osg::Drawable*`（自定义 `LaserCloudDrawable`） |

### 输入数据来源与含义

渲染器通过 `LaserCloudFuseDeviceContext` 借裸指针直接读融合算子的常驻显存，**不持有任何点云副本**：

```
激光08 laser_reconstruct (CUDA)
  │  单帧激光3D点云, 相机坐标系, GpuMat (CV_32FC3)
  ▼
融合-02 LaserCloudFuseCuda::Execute(d_points3d, R, T)
  │  gx = R·p + T → floor(g/voxelSize) → atomicCAS 体素哈希去重
  │  → 跨帧累积的"代表点"写入 d_fusedXyz_（每体素1个点, 全局坐标系）
  ▼
融合-03 LaserCloudNormalCuda::Execute(fuse, ...)
  │  新建体素邻域协方差 + Jacobi 特征分解 → 写回 d_fusedNormal_
  ▼
fuse.GetDeviceContext()  →  LaserCloudFuseDeviceContext (POD)
  ▼
LaserCloudRenderer::update(ctx)   ← 仅存指针, 零拷贝
```

`LaserCloudFuseDeviceContext` 的 8 个字段中，渲染器仅使用 4 个：

| 字段 | 类型 | 渲染器使用 | 含义 |
|------|------|:--------:|------|
| `d_fusedXyz` | `const float*` | ✅ | 代表点坐标（SoA: x0,y0,z0,x1,...），全局坐标系 |
| `d_fusedNormal` | `const float*` | ✅ | 代表点法线（同布局），由融合-03 写入 |
| `voxelSize` | `float` | ✅ | 体素边长，展点 halfSize = voxelSize×0.5 |
| `fusedPointCount` | `size_t` | ✅ | 当前累积代表点数 |
| `d_keys` | `const uint64*` | ❌ | 体素哈希键（融合内部去重用） |
| `d_fusedIdx` | `const uint32*` | ❌ | 哈希槽→代表点索引（融合内部用） |
| `mask` | `uint64` | ❌ | 哈希表掩码（融合内部用） |
| `invVoxelSize` | `float` | ❌ | 体素倒数（融合内部用） |

> **什么是"代表点"**：融合算子在 GPU 维护一张体素哈希表（默认 0.5mm 格、2M 槽）。每帧 ~5 万个激光点经 R/T 变换到全局系后量化到体素格，`atomicCAS` 首个占据某体素的点成为该体素的"代表点"写入 `d_fusedXyz`，后续落入同体素的点仅计数不写入。扫 100 帧后 `d_fusedXyz` 中是去重累积后的几十万个全局坐标点。
>
> **零拷贝**：`d_fusedXyz` / `d_fusedNormal` 已在融合算子的 GPU 显存中（跨帧持久）。展点 kernel 直接读该显存，写入 CUDA-GL mapped VBO，全程 GPU 内部，零 PCIe 传输。渲染器每帧 `update()` 仅传 4 个标量/指针值，无数据搬运。

---

## C. 算法

**核心流程**（每帧 `drawImplementation` 中执行）：

```
Step 1: CUDA-GL map — 将 3 个 GL VBO 映射为 CUDA 可写指针
Step 2: 展点 kernel — 每点一个 CUDA 线程：
        a. 读 d_fusedXyz[i] (位置) + d_fusedNormal[i] (法线)
        b. 构建正交基 (T=切线, B=副切线) from 法线
        c. 生成 4 个四边形顶点：P ± T×halfSize ± B×halfSize
        d. 写入 mapped VBO (pos + uv + normal)
Step 3: CUDA-GL unmap — CUDA 写入对 GL 可见
Step 4: glDrawElements — 渲染 N×2 个三角形
```

**片元着色器（正反面区分）**：

```glsl
if (gl_FrontFacing) {
    // 正面：蓝灰色 (0.25, 0.55, 0.85) + 法线光照
} else {
    // 背面：暗灰色 (0.35, 0.35, 0.38) + 弱光照
}
```

**关键函数/技术**：

| 函数/技术 | 用途 |
|-----------|------|
| `cudaGraphicsGLRegisterBuffer` | 注册 GL VBO 为 CUDA 资源（`WriteDiscard` 模式） |
| `cudaGraphicsMapResources` | 每帧映射 VBO，获取 CUDA 可写指针 |
| `launchExpandLaserPoints` | 展点 kernel 启动器（3 个 SoA 输出：pos/uv/normal） |
| `buildOrthonormalBasis` | 从法线构建切平面坐标系（device 内联函数） |
| `osg::Drawable` 子类 | 绕过 OSG VBO 管理，直接控制 GL buffer + draw call |
| `osg::GLExtensions` | 通过 OSG 获取 GL 函数指针（glGenBuffers 等） |
| `gl_FrontFacing` | 片元着色器区分正反面颜色 |

---

## D. 依赖

**上下游算子**：

```
laser_cloud_fuse_cuda (融合-02) → d_fusedXyz + d_fusedNormal
laser_cloud_normal_cuda (融合-03) → d_fusedNormal (填充法线)
    ↓
[本算子] LaserCloudRenderer → osg::Drawable → OSG 场景图 → 屏幕
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| `LaserCloudFuseDeviceContext` | 来自融合算子，含 `d_fusedXyz` / `d_fusedNormal` / `voxelSize` / `fusedPointCount` |
| GL VBO（3 个 + 1 索引） | 常驻显存，CUDA-GL 共享。首帧创建，跨帧复用 |
| `point_expand_kernel.cu` | CUDA 展点 kernel（正交基数学） |

**头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `laser_cloud_fuse_cuda.h` | `LaserCloudFuseDeviceContext` 定义 |
| `point_expand_kernel.h` | 展点 kernel 启动函数 |
| `common/calib_logging.h` | 日志宏 |
| `<osg/Drawable>` / `<osg/GLExtensions>` | OSG Drawable 基类 + GL 函数指针 |
| `<cuda_gl_interop.h>` | CUDA-GL 互操作 API |

> **模块独立**：不依赖 `marker_cloud_renderer` 或 `point_expand_math.h`，可单独使用。

---

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| `LaserCloudFuseCuda::GetDeviceContext()` | `LaserCloudFuseDeviceContext`（值传递，POD） | 含 GPU 指针，零拷贝 |

**本算子→下游**：

| 输出字段 | 传递给 | 传递方式 |
|---------|--------|---------|
| `getDrawable()` | `osg::Geode` | `osg::Drawable*`（ref_ptr 托管） |

**调用示例**：

```cpp
ScannerViewer viewer(1 << 20);  // 100 万点
viewer.init(1280, 720);

// 每帧（扫描中）：
auto ctx = fuse.GetDeviceContext();
viewer.laserRenderer()->update(ctx);  // 仅存储指针
viewer.frame();                        // GPU 零拷贝展点 + 渲染
```

---

## E. 架构

**文件结构**：

```
display/
├── laser_cloud_renderer.h          # 公开头文件（LaserCloudRenderer 类）
├── laser_cloud_renderer.cpp        # LaserCloudDrawable + LaserCloudRenderer::Impl
├── point_expand_kernel.h           # kernel 启动函数声明
├── point_expand_kernel.cu          # CUDA kernel 实现（展开数学）
└── point_expand_math.h             # 共享数学（ExpandVertex + expandMarkerPointCPU，本算子不直接依赖）
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `LaserCloudRenderer`（pImpl，有状态） |
| 内部 Drawable | `LaserCloudDrawable : osg::Drawable`（自定义，管理 GL VBO + CUDA interop） |
| 更新方法 | `update(LaserCloudFuseDeviceContext&)` — 仅存储 GPU 指针 |
| 获取节点 | `getDrawable()` → `osg::Drawable*` |
| 重置 | `clear()` |
| 日志标签 | `"Display-LaserCloudRenderer"` |

**VBO 布局**（3 个独立 VBO + 1 静态索引）：

| VBO | 格式 | 大小（1M 点） | 写入者 |
|-----|------|-------------|--------|
| positions | float×4N×3 | 48 MB | CUDA kernel |
| uvs | float×4N×2 | 32 MB | CUDA kernel |
| normals | float×4N×3 | 48 MB | CUDA kernel |
| indices | uint×N×6 | 24 MB | CPU 一次性生成 |

---

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenSceneGraph | 3.6.5 | `osg::Drawable` / `osg::GLExtensions` / `osgViewer` |
| CUDA Toolkit | ≥ 12.x | `cudaGraphicsGLRegisterBuffer` 等 interop API |
| OpenGL | ≥ 3.3 | VBO + vertex attrib + shader + `gl_FrontFacing` |
| opengl32.lib | — | Windows GL 运行时（`glDrawElements` 等） |

---

## F. 参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `maxPoints` | size_t | 无默认值（ScannerViewer 传入 1<<20） | 最大点数，决定 VBO 预分配大小 |

**显存占用**：`maxPoints × (48+32+48+24) MB / 1M = 152 MB`（1M 点时）

---

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 渲染目标 | 30+ FPS（实测 153 FPS，1M 点） |
| GL context | 须在 OSG `realize()` 后使用（首帧自动初始化 VBO） |
| 线程模型 | OSG `SingleThreaded`（CUDA-GL 同步依赖同线程） |
| VBO 生命周期 | 跨帧持久，`releaseGLObjects()` 时释放 |
| 法线要求 | 须归一化（kernel 内部 rsqrtf 保证） |
| 正反面 | 片元着色器区分：正面蓝灰、背面暗灰 |

---

## K. 质量

| 测试项 | 结果 |
|--------|------|
| 100 万点端到端 | 6.5 ms / 153 FPS |
| 展点 kernel | ~3 ms |
| 光栅化 200 万三角形 | ~4 ms |
| PCIe 传输 | 0 ms（零拷贝） |

---

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | GL VBO 在首帧 `drawImplementation` 中延迟创建 | 首帧多 ~80ms（一次性） |
| 🟢 低 | `cudaGraphicsMapResources` 隐式同步可能阻塞 GL | 实测影响 <0.1ms |
| 🟡 中 | OSG `Drawable` 克隆方法返回空对象（不支持深拷贝） | 场景图中不应克隆此 Drawable |
| 🟡 中 | CUDA-GL interop 上下文中访问独立 device 指针可能冲突 | cull kernel 暂未启用（153 FPS 无需剔除） |

---

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
| **现有模块** | `display/laser_cloud_renderer`（`LaserCloudRenderer` + `LaserCloudDrawable`） |
| **优化历程** | V1 (D2H+CPU 填充, 9 FPS) → V1+pinned+memcpy (25 FPS) → V2 零拷贝 (153 FPS) |
| **模块依赖** | 独立于 `marker_cloud_renderer`，仅依赖 `point_expand_kernel` + 融合算子 |

---

> **文档结束**
