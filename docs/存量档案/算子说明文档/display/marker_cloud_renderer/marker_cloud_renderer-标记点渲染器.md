# 标记点渲染器

> 实现位置：`modules/03_rendering/display/`（并入 `mod_rendering`）。本组件为渲染组件（OSG/CUDA-GL interop），非流水线算子，不适用算子规范 §2 三元组/Execute 契约。

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 显示-02 |
| 中文名称 | 标记点渲染器 |
| 英文目录名 | marker_cloud_renderer |
| 运行平台 | CPU + OpenGL（GPU） |
| 所属流程 | 显示渲染流程（独立流程，消费融合-02M 输出） |
| 精度档次 | —（渲染，非测量） |

---

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 融合-02M 输出的全局标记点集（已 R/T 变换 + 去重） | `const std::vector<MarkerCloudPoint>&` |
| **输出** | OSG Geometry 节点（白圆 + 黑环 SDF 渲染） | `osg::Geometry*` |

> **数据来源**：`MarkerCloudPoint` 由 `marker_cloud_fuse_cpu`（融合-02M）输出：
> - `x/y/z` — 全局坐标系位置（已 R/T 变换）
> - `nx/ny/nz` — 全局坐标系法线（已 R 旋转）
> - `whiteRadius` — 白色标记点半径

> **零拷贝引用**：`update()` 仅存储 `const std::vector<MarkerCloudPoint>&` 引用，不复制数据。渲染器自身不做 R/T 变换和去重。

---

## C. 算法

**CPU 展点 + GPU SDF 渲染**：

```
Step 1: update(fusedPoints) — 存外部引用 (~0us)
        a. 存储 const vector<MarkerCloudPoint>& 指针
        b. 记录点数，不做拷贝

Step 2: flush() — 展开为四边形 + 上传
        a. expandMarkerPointCPU() 展开每点为 4 顶点
        b. outerRadius = whiteRadius × 10/6（黑环外半径）
        c. 填充 OSG 数组 + dirty()

Step 3: Fragment Shader SDF（GPU）
        正面 (gl_FrontFacing):
          dist = length(v_uv)
          if dist <= 0.6  → 白色圆
          else if dist <= 1.0 → 黑色环
          else → discard
        背面:
          if dist <= 1.0 → 暗灰色圆 (粘胶面)
          else → discard
```

**黑环比例**：外径 = 白色半径 × 10/6 ≈ 1.667 × 白色半径

**关键函数/技术**：

| 函数/技术 | 用途 |
|-----------|------|
| `expandMarkerPointCPU` | CPU 端正交基展开（来自 `point_expand_math.h`，header-only） |
| SDF 圆形 | Fragment shader 程序化绘制白圆 + 黑环，无需纹理 |
| `gl_FrontFacing` | 正反面区分：正面白圆黑环、背面暗灰 |
| `MarkerCloudPoint` | 融合-02M 输出的全局标记点结构（位置 + 法线 + 半径） |

---

## D. 依赖

**上下游算子**：

```
marker_cloud_fuse_cpu (融合-02M) → GetFusedPoints()
    ↓ vector<MarkerCloudPoint>&
[本算子] MarkerCloudRenderer → osg::Geometry → OSG 场景图 → 屏幕
```

**头文件依赖**：

| 头文件 | 用途 |
|------|------|
| `core/marker/marker_cloud_fuse_cpu/marker_cloud_fuse_cpu.h` | `MarkerCloudPoint` 结构体 |
| `point_expand_math.h` | `ExpandVertex` + `expandMarkerPointCPU`（共享数学，header-only） |
| `<osg/Array>` / `<osg/PrimitiveSet>` | OSG 数组 + 索引缓冲 |

> **模块独立**：不依赖 `laser_cloud_renderer`，仅依赖 `point_expand_math.h` 和 `MarkerCloudPoint`。

---

## D2. 衔接

**调用示例**：

```cpp
MarkerCloudFuseCPU fuse;  // 融合-02M，全局标记点唯一权威源
MarkerCloudRenderer renderer(1024);

// 每帧：
auto result = fuse.Execute(frameMarkers, R, T);     // 融合：R/T变换 + 体素哈希去重

renderer.update(fuse.GetFusedPoints());    // 渲染器：存引用，零拷贝
renderer.flush();                          // 展点 → VBO
viewer.frame();                            // OSG 渲染
```

---

## E. 架构

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `MarkerCloudRenderer`（pImpl，有状态） |
| 更新数据源 | `update(const vector<MarkerCloudPoint>&)` — 存储外部累积结果引用 |
| 展开上传 | `flush()` — CPU 展点 + OSG VBO 上传 |
| 获取节点 | `getGeometry()` → `osg::Geometry*` |
| 清空 | `clear()` — 置空引用 + 隐藏渲染 |
| 点数查询 | `pointCount()` |
| 日志标签 | `"Display-MarkerCloudRenderer"` |

**内部结构**：

```cpp
struct Impl {
    const std::vector<MarkerCloudPoint>* srcPoints_ = nullptr;  // 外部数据源引用
    size_t srcCount_ = 0;

    // 预分配的 OSG 固定 VBO（maxMarkers × 4 顶点）
    osg::ref_ptr<osg::Vec3Array>        positions_;
    osg::ref_ptr<osg::Vec2Array>        uvs_;
    osg::ref_ptr<osg::Vec3Array>        normals_;
    osg::ref_ptr<osg::DrawElementsUInt> indices_;
};
```

---

## F. 参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `maxMarkers` | size_t | 1024 | 最大渲染标记点数 |
| `RING_RATIO` | float (constexpr) | 10/6 | 黑环外径 / 白圆半径 |

---

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 标记点数量 | 典型 <100/帧，最大 1024 |
| update 耗时 | ~0 us（仅存引用） |
| flush 耗时 | 500 点 ≈ 27 us（CPU 展点 + VBO 上传） |
| 法线来源 | `marker_cloud_fuse_cpu` 输出的全局坐标系法线（point_reconstruct → R变换） |
| 正反面 | 正面白圆+黑环、背面暗灰圆 |
| NodeMask | 空 Geometry（0 索引）天然不渲染，`flush()` 填充顶点后显示 |
| 数据所有权 | **不拥有数据**，`update()` 传入的 vector 在 `flush()` 期间必须有效 |

---

## K. 质量

| 测试项 | 结果 |
|--------|------|
| update(500 点) | ~0 us（存引用） |
| flush(500 点) | ~27 us |
| 标记点全路径 (fuse 25帧 + update + flush) | ~0.1 ms |
| SDF 白圆+黑环 | 视觉正确，正反面区分 |

---

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
| **现有模块** | `display/marker_cloud_renderer`（`MarkerCloudRenderer`） |
| **模块依赖** | 依赖 `marker_cloud_fuse_cpu`（MarkerCloudPoint 结构体）和 `point_expand_math.h` |

---

> **文档结束**
