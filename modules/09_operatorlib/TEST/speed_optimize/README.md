# scan_replay 扫描链回放测试 · 使用说明

> **一句话**：把真机录好的扫描照片（左右相机各 201 张）从头到尾过一遍扫描算子流水线，边跑边"截图＋掐表＋判卷"——产出设备位姿 R/T、标记点 3D、激光 3D 点云，并逐步留证据。
>
> 对应流程 `docs/流水线/客户端扫描流水线.md` §一；设计文档 `docs/plans/2026-10-04-scan-chain-replay-test-design.md`（v2.5）。

---

## 0. 目录结构与归档说明（2026-10-22 整理）

```
speed_optimize/
├── README.md / CMakeLists.txt / config*.json   # 说明＋构建＋活动配置
├── ops/ src/ tests/ cmake/ compare/            # 代码与工具（位置不动）
├── golden/ date_in/                            # 对拍基线＋输入数据（活动资产）
└── 归档/
    ├── 记录文档/    # 31 份历史会话/设计/实验/实施记录 md（含最新接力：
    │                 # 交接会话-冷启动复测与七处修复-2026-10-22.md）
    └── 实测数据/    # 151 个历史 out_*/ab_*/p7* 等实测产物目录
                      #（含原散落在仓库根目录的 50 个 out_*，2026-10-22 收编）
```

- 历史记录文档与实测数据已全部移入 `归档/`，顶层只保留构建与活动资产
- 查最新进度/接力：`归档/记录文档/` 下按文件名日期取最新交接会话
- 新实测产物建议直接 `--dump-dir 归档/实测数据/<名字>` 或事后移入，避免顶层再膨胀

---

## 1. 能干什么

| 你想做的事 | 行不行 |
|---|---|
| 验证 16 个扫描算子串起来在真实数据上跑得通 | ✅ 主用途 |
| 查看每步算子"到底算出了什么"（图片/点表/3D 点云） | ✅ 逐步可视化 |
| 查每个算子耗时（谁慢） | ✅ 逐算子计时 |
| 单独调某个算子参数（只重跑那一步） | ✅ 单步模式 |
| 验证扫描精度是否达标 | ⚠ 只能看统计，精度受标定数据质量限制（见 §8） |
| 实时接相机扫描 | ❌ 这是回放工具，不是实采 |

## 2. 先懂两个概念（不懂会误用）

**① 每张图都同时有激光和标记点（同帧）**

扫描仪边运动边扫。每张照片里既有**激光线**（测物体表面形状）也有**标记点**（反光圆点，测设备位姿）。设备在动，**不能用两张照片拼一个结果**（第二张时位置已经变了）——所以每张照片独立跑完整条链：激光线→激光 3D，标记点→R/T，**两者出自同一瞬间、同一位姿**。

**② 奇数帧/偶数帧＝两支激光管轮流发射**

奇偶不是"有激光/没激光"（每张都有！），而是**本次是哪支管打的**——两支管要用不同标定参数。工具按帧号奇偶自动切换参数表（第二支管的标定文件在 config 预留接入位，目前两管共用一张表）。实测两管产出差异明显（奇帧少、偶帧多属正常）。

## 3. 跑之前要有什么（环境＋输入要求）

### 3.1 硬件与构建

| 条件 | 说明 |
|---|---|
| Windows＋NVIDIA 显卡 | CUDA 算子需要 |
| Release 构建 | Debug 版 OpenCV 缺 imgcodecs→本工具只在 Release 存在（Debug 构建自动跳过不报错，Debug 基线不受影响）；需 OpenCV 全组件（imgcodecs＋cudaarithm＋cudafilters） |

```powershell
# 首次构建
cmake -S . -B build-rel -G "Visual Studio 17 2022" -A x64 -DCMAKE_CONFIGURATION_TYPES=Release -DJMW_GH_MIRROR=https://ghfast.top/
cmake --build build-rel --config Release --target scan_replay test_scan_chain_replay
```

### 3.2 帧数据目录结构（唯一有硬性结构的输入）

```
<dataDir>                ← config paths.dataDir 指向（任意位置）
├── left/
│   ├── 000000.png       ← 6 位零填充命名（恰 10 字符，其余文件被忽略）
│   └── ...
└── right/
    ├── 000000.png       ← 与 left 同序号一一对应（缺帧报错）
    └── ...
```

| 项 | 要求 | 不满足的后果 |
|---|---|---|
| 目录 | 必须含 `left/`、`right/` 两个小写子目录 | 启动报错退出 |
| 命名 | `NNNNNN.png`（6 位零填充，恰 10 字符） | 非 10 字符文件被忽略 |
| 数量 | 左右目同序号一一对应 | 缺帧报错 |
| 格式 | PNG 灰度（可读为 CV_8UC1） | 该帧读取失败 |
| **内容语义** | **每帧同时含激光线＋标记点**（同帧模式）；奇偶＝激光管号 | 结果无意义 |
| 尺寸 | 与标定文件 `imageSize` 一致（当前 2048×1536） | 矫正结果错位 |

### 3.3 标定文件（位置无结构要求，路径可指任意处，不必与帧同目录）

| 输入 | 要求 |
|---|---|
| `cameraCalib`（JSON） | factory_calib 模块1 输出 schema，必须含四节：`intrinsic`（left/right 的 `camera_matrix` 3×3＋`dist_coeffs` 5 参数，snake_case）／`extrinsic`（`R` 3×3＋`T` 3）／`stereoRectifyTempTable`（各档含 `temperature`＋`R1/R2/P1/P2/Q`＋`compensatedCameraMatrixL/R`＋`compensatedT`）／`imageSize`。名义温在档中**就近选档**（当前表 7.5~37.5℃/0.5 步距 61 档） |
| `cmttSidecar`（.bin） | CMTT 二进制温表（96B 头＋逐档索引＋CMPU blob），按名义温**就近取单档**（~38MB 不整载）；装载时校验 rowStep 与 `epipolar_interp_dual.epipolar_row_step`(0.7) 一致——相等 PASS／倍数 WARN／无关系报错 |
| `cmttSidecarEven`（可选） | 第二激光管标定分立时填（留空＝两管共表） |

### 3.4 路径写法规则（config `paths` 节）

| 写法 | 解析为 |
|---|---|
| `{JMW_SOURCE_ROOT}/xxx` | 工程根展开（推荐，可移植） |
| `E:/绝对/路径` | 原样使用 |
| `相对路径` | 以**工程根**为基准（不以 exe 或 cwd） |

数据自包含于 **`test_data/`**（2026-10-05 起，gitignore 不进版本库）：`20260831_171343_even/`（78 帧左右目）＋`camera_calib.json`＋`curve_map_temp_table.bin`（2.25GB CMTT sidecar）＋`laser_calib.json`（存档参考，链路不消费）。config `paths` 默认指向本地副本；换数据集＝改 `paths` 三个路径（或指回 `factory_calib/` 原址均可）；目录只需读权限，产物另走 `outDir`。

### 3.5 隐含一致性要求（代码不显式校验，错了结果静默变差）

- 标定与帧出自**同一台设备**同一标定会话
- 名义温度取采集时实际温度附近（敏感性可用 `--temp-compare` 探查）
- 帧序号奇偶与激光管发射顺序一致（管号仅用于选标定表，两管共表时无影响）
- 真实数据缺席时：gtest 自动 SKIP、CLI 报错退出码 1

## 4. 快速上手

```powershell
$env:PATH = 'C:\opencv-cuda-4.13.0\x64\vc17\bin;' + $env:PATH
$exe = '.\build-rel\modules\09_operatorlib\TEST\scan_chain_replay\Release\scan_replay.exe'

& $exe --cycles 3     # 只跑前 3 帧，几秒出结果
```

跑完看两处：① 屏幕每帧一行 `[cycle N] OK markers=10/4 reg=flow laser=10616/10616 rowResid=0px`（2026-10-06 conic 链口径：检出/有效、配准源、激光 匹配/3D）；② `out\cycle_000\` 全部中间产物（见 §5）。

```powershell
& $exe --cycles 0                        # 全量 201 帧（约 8 分钟）
& $exe --step zernike_edge --cycles 1    # 只看某步产物（调参用；--list-steps 列全部步骤名）
& $exe --temp-compare 22.5,25.0          # 两个温度各跑一遍对比
& $exe --dump-dir D:\tmp\scr             # 产物目录覆盖；--help 看全部参数
```

退出码：0 正常｜1 配置/数据/标定错误｜2 过半帧失败。

> **⚠️ 性能测量纪律（2026-10-23 错配事故后立，必读）**：
> ①跑性能基线一律用 **config_flat ＋ `--data-dir .../laser-mark1688-1006/ALL_raw` ＋ `JMW_FIVECHAIN=1` ＋ `--lean`**——
> **config_all 是 steger 旧提取器路径，禁止用于性能对比**（帧率天然低 20-30%）；
> ②~~性能跑批前设 JMW_OCV_THREADS=0~~ **已过时（2026-10-24）**：B1 帽已翻转落地——现默认
> `setNumThreads(0)` 纯顺序（最优档），**无需再设任何环境变量**（`JMW_OCV_THREADS=1`＝对照臂）；
> ③速度结论只认同配方 ABBA 交替，单轮不可引用（机器自身波动 ±10 FPS 在案——绝对帧率带
> 142-158 皆正常，看「新旧同场对比差值」而非绝对数）。
> 详见 `归档/记录文档/会话记录-2026-10-23-配置错配与B1线程帽复盘.md` §七勘误。

## 5. 结果怎么看（产物对照表）

产物在 `out\`（config 可改）。可视化**全部左右并合一张图**（L|白隔条|R），命名：**0-＝输入原图｜L\*＝激光链（L0 掩膜→L5 3D）｜M\*＝标记点链（M0 掩膜→M11 配准）**：

| 文件 | 步骤 | 是什么 |
|---|---|---|
| `0-frames.png` | 输入 | 原始左右灰度帧并合 |
| `0-binary.png` | 分离前端 | 高斯5×5＋二值化(threshold)后**未分割**的全量前景（调 threshold 看这张） |
| `L0-laser_mask.png` | 分离→激光链 | 激光掩膜（白=掩膜，纯二值）L\|R |
| `M0-marking_mask.png` | 分离→标记点链 | 标记点掩膜（纯二值）L\|R |
| `L1-steger.png`＋`L1-steger_L.csv` | 激光链① | 原帧＋绿色激光中心点 |
| `L2-undistort_cuda.png` | 激光链② | 矫正后激光点（暗画布，矫正系坐标） |
| `L3-interp.png`＋`L3-interp_L/R.csv` | 激光链③ | 原帧＋黄色插值点 |
| `L4-match_v3.png`＋`L4-match_L/R.csv` | 激光链④ | **白=未匹配；彩色=匹配对（按激光线号同色同编号、编号与线同色，左右对应）** |
| `L5-laser3d.ply` | 激光链⑤ | 激光 3D 点云（MeshLab/CloudCompare） |
| `M1-ccl.png` | 标记点链① | 组件伪彩标签＋白框＋编号 |
| `M2-split.png` | 标记点链② | ROI 子图网格拼贴 L\|R |
| `M3-zernike.png`＋`M3-zern_L.csv` | 标记点链③ | ROI 拼贴＋红色边缘点 |
| `M4-merge.png` | 标记点链④ | 原帧＋边缘点按组着色 |
| `M5-undistort_cpu.png` | 标记点链⑤ | L/R 矫正点同图散点（蓝/红）——**行对齐证据图** |
| `M6-ellipse.png`＋`M6-ellipse_L/R.json` | 标记点链⑥ | 椭圆轮廓＋中心十字＋组号 |
| `M7-marker_match.png` | 标记点链⑦ | 中心配对：**匹配对同色同编号（编号=左索引）**、未匹配灰；右图逐对标注 `dy=±x.xx`（yL−yR 离极线行偏差）＋ `d=xxx.x`（xL−xR 视差），与编号同色 |
| `M10-markers3d.ply` | 标记点链⑧（conic_reconstruct；文件名沿用 M10 前缀） | 圆锥相交重建的标记点圆环（每有效标记 36 环点＝ringPointsForViz；MeshLab/CloudCompare） |
| `M11-registration.json`＋`M11-trajectory.png`（out 根） | 配准 | 配准详情＋全程 T 的 XZ 轨迹 |
| `timing.csv`／`report.csv`／`consistency.json`／`cycle_NNN/summary.json` | 汇总 | 逐帧耗时／逐帧统计（钉阈值数据源）／双链标定一致性／逐帧摘要（内含 `conicHistogram`＝conic_reconstruct 拒因直方图，枚举名→计数、None＝过闸对计数，判读见 §6） |

⚠ 只有前 3 帧落全量产物（`dumpFullFrames=3` 防磁盘爆），其余帧只有 summary；想看某帧全量就 `--cycles N` 单跑。计时是"每步等 GPU 干完"的口径——单步耗时真实，整链总耗时不代表产线速度。

## 6. 怎么判断跑对了

| 指标 | 正常 | 异常→可能病因 |
|---|---|---|
| `[cycle N] OK` | 绝大多数 OK | DEGRADED 多→看 stderr `step xxx FAIL` 行定位 |
| `markers=检出/有效` | 检出 ~10/帧（v3 实测 8-12，真实标记点为主光晕环少量混入）；有效中位 5（conic 空间圆口径） | 检出 0→标记点掩膜空（查 §7 分档参数 noiseErode/laserKill）；有效长期 0→看 `cycle_NNN/summary.json` 的 `conicHistogram` 拒因——NoRealRoot/VirtualPlane/NotCircle/Inconsistent 主体＝本批标定质量代偿所致（工作点见 §8），Radius 多＝半径闸过紧 |
| `laser=匹配/3D` | ~1.1 万/帧（v3 温表全表注入后 8651-13923，匹配对全量转 3D） | 恒 0→激光掩膜空或温表选档错 |
| `rowResid` | 0px | 变大→行对齐坏（标定退化） |
| 一致性 max\|Δ\| | 0 | >1e-6→标定注入链路断（工具 bug 级） |

自动判卷（16 用例＝15 PASS＋1 既有 FAIL〔StatsGates：数据漂移，与去重无关，待重钉〕；数据缺席 SKIP）：

```powershell
ctest --test-dir build-rel -C Release -R test_scan_chain_replay
```

### 6.1 去重双路径（runtime.dedupPath）与真实数据 A/B 结论

> 2026-10-13 顺带修复：CPU oracle `steger_dedup_merge.cpp` `apply()` 先 move 后读 `out.size()`（MSVC moved-from vector 为空）致 cpu 路径 `totalPointCount` 恒 0——Parity 恒等式桥（cpuMerged=N−cpuOut）依赖此值，随 commit 3ae1b42 修复。

steger 出点后的平顶去重有两条路径，config `runtime.dedupPath` 切换（设计文档 `docs/plans/2026-10-12-steger-fast-gpu-dedup-design.md`；v2 重设计 `docs/plans/2026-10-13-steger-dedup-v2-design.md`）：

| 路径 | 值 | 是什么 |
|---|---|---|
| 算子内 GPU 去重（默认） | `operator` | steger_fast `dedupMode=1`（config 模板已置 1；2026-10-13 v2：位移式六步 D1-D6 全在 GPU 算子内完成），零外挂开销——生产默认路径 |
| 外挂 CPU 位移式合并 | `cpu` | `steger_dedup_merge`（kNN-PCA 切向）：**对照 oracle** 用；harness 同时强制算子 dedupMode=0，防双重去重。CLI 运行看 stdout 的 `[dedup:operator]`/`[dedup:cpu]` 行确认生效 |

**真实数据 A/B 实测（2026-10-13 v2）**：GPU=operator 路径，CPU=外挂 oracle 路径；**gtest 对齐用例跑 laser-mark1688-1006 全量目录（前 3 cycle），CLI A/B 用 sparse_30 子集（前 3 帧）**：

| 指标 | GPU（算子内 v2） | CPU（外挂 oracle） | 对齐 |
|---|---|---|---|
| 每帧合并点数 L/R（sparse_30 前 3 帧） | **L 4031-4184 / R 4265-4401** | ~4.0k / ~4.4k | **恢复 CPU 量级**（v1 仅 L 5-7/R 4-7 差 3 个数量级——已消除） |
| 探针 gpu/uf 比值 | 0.9995-1.0010 | — | ≈1 |
| gtest Parity merged（3 cycle L+R Σ） | 25219 | 25223 | **比值 0.9998** |
| 下游 v3 matched / laser3D | — | — | **差 ≤1 点（rel≈1e-4）** |

**结论：已修复**。v1 双病因（① 真实平顶上 Hessian 法向混沌；② 主损失源＝GPU 骨架绝对坐标投影在 |p|≈900px 图幅放大切向微角差成沿差切段）→ v2 双改（① 邻居 PCA 几何切向 GPU 化；② 位移门配对判据 Δp·t̂ᵢ 原点无关＋标签传播成簇），GPU 合并量收敛到 CPU 结构。对齐由 gtest 用例 `DedupAlign.ParityOperatorVsCpu` 常驻把关，两步钉键：`runtime.check.dedupAlignMergedRatioMin=0.9`（实测 0.99992）／`dedupAlignDownstreamDiffMax=0.01`（实测 5.9e-5）。v1 限界历史与归因证据存档于算子说明 §F2（探针 `--skeleton` 模式，沙箱未提交）。

### 6.2 激光中心提取双路径（runtime.extractor，2026-10-19）

`runtime.extractor` 切换激光中心提取算子，两路共存可随时对拍：

| 路径 | 值 | 是什么 |
|---|---|---|
| 现役 Steger（默认） | `steger` | `steger_fast`（Hessian 脊线＋dedupMode 去重）——生产默认路径，零改动 |
| 平顶专用备用 | `flat` | `flat_center`（`ops/core/laser/flat_center`，设计文档 `docs/plans/2026-10-18-flat-center-extractor-design.md`）——逐列切段→剖面分类（平顶→**背景基准半高中点**/尖峰→去背景重心）＋统一自检门，**每段恰出 1 点** |

**为什么做**：真实激光线为平顶剖面。本数据集金样实证（`out_golden_base` L1r==L1c=33638 点）——steger 去重在本数据 **一个双点都没并**，每线每列以 dy=1 双肩对出点（17,477 对/眼/帧）；flat 逐段单点从结构上消灭双点。设计期数字见 README §6.1 表（sparse_30 数据集去重生效场景）。

**实测（2026-10-19，本数据集前 3 帧，gtest `ExtractorParity`）**：steger 出点 197,613（折叠双点后 106,952）｜flat 出点 94,454（＝折叠口径的 88.3%，自身段覆盖 98%，弃 324/16,256 段）｜下游 matched 差 4-8%（单点喂链 vs 双点喂链的结构性差异，10% 观察门内）｜速度 **flat 2.81ms/眼 vs steger 3.32ms/眼（1.18×）**。

**实现要点**：单 kernel 逐列逐段两次行走＋括号重心自检；**列带切分 ×8**（1 线程/列在 1400 行图上延迟饥饿，实测 7.8ms→2.8ms）；半高＝`vmin + halfRatio×(vmax−vmin)`（真实帧背景 ≈110-120，绝对半高只比背景高 ~12 灰度级会找不到沿——链级对拍实测修正）；六门弃点（minRun/contrast/bright/edge/gate/width）→ Degraded＋按门计数消息。约束：与 `dedupPath=cpu` 互斥（config 硬校验）；ByLabel 模式为标定链预留（本链只走 Flat）；步骤槽位键恒为 `steger_fast`（下游/报表/导出零改动，flat 时 summary 多 `extractor` 字段）。窄高斯 σ2 对拍被排除（steger 自身伪响应，flat 精度由单测独立验证）。


## 7. 参数在哪调、哪些不能动

全部参数在同目录 `config.json`，**每个参数都有中文注释**（`_comment_` 键）：

| 想调什么 | 改哪 |
|---|---|
| 跑几帧 | `runtime.maxCycles`（或 `--cycles`） |
| 名义温度（标定表按温选档） | `runtime.nominalTemperature` |
| 某算子算法参数 | `operators.<算子名>` 节（如 `operators.conic_reconstruct`；注意 marker_match 是下划线风格 `y_tolerance`） |
| conic 代偿工作点 | `operators.conic_reconstruct` 的 `rowOffsetPx`/`circleAxisRatioMax`（能调但换标定后必须复位，见 §8） |
| 断言阈值（注释含实测依据） | `runtime.check` |
| 第二支激光管的标定表 | `paths.cmttSidecarEven`（留空＝两管共表） |
| 去重路径 | `runtime.dedupPath`（`operator`＝算子内 GPU 去重【默认】/`cpu`＝外挂 CPU 对照 oracle；说明见 §6.1） |
| 激光中心提取算子 | `runtime.extractor`（`steger`＝现役【默认】/`flat`＝flat_center 平顶专用；说明见 §6.2） |
| 光流误差门 | `operators.optical_flow_fuse.rmseMaxGate`（0=关【旧口径】；3.0＝推荐，解算误差超门整帧降级——2026-10-14 新增） |
| 法线可信度加权 | `operators.optical_flow_fuse.useNormalWeight`（**2026-10-15 裁决默认关**：等权解算；朝向门放宽到 30°+ 时置 true 作软降级保险丝） |
| 新点发号查重半径 | `runtime.mintDedupRadiusMm`（默认 5.0；发新号前先对地图查重防重复编号污染；独立于匹配门，须远小于标记点间距） |
| 误差趋势门 | `runtime.flowRmseDriftRatio`（默认 10.0、0=关；误差超"前 50 成功帧中位×倍率"判失败——抓温漂/斜视角等系统性慢变） |
| 首帧建系最低点数 | 算子内置 ≥max(minMatchedPoints,4)（不可配；防 3 点零冗余盲区，2026-10-15） |
| 规则网格板 | 自动检测并**硬拒**（最近邻间距高度一致＝规则阵列；换用随机布点板，2026-10-15） |
| 标记点板布点要求 | **随机/非周期贴点**（规则网格＝平移对称原理性不可分辨）；换板时门限须满足 `matchDistThresh ≤ 最小间距÷10` |
| 配准失败诊断 | 无需配置——summary.json 逐帧含 `flowDiag`（matched/rmse/message）与 `relock`（尝试结果） |

**⚠ 不能动的（harness 内部约定，config 里没有也不该有）**：相机 K/D、矫正矩阵等标定派生参数（自动从 `camera_calib.json` 读，手改会两份数据打架；conic_reconstruct 的 K1/K2/C2 同理由温表档 P1/P2 派生注入，不进 config）；右目 P 矩阵平移项置零与宽圆斑组合掩膜（正确性约定，见 §9-2）。

**⚠ 能调但换标定后必须复位的代偿工作点**：conic_reconstruct 的 `rowOffsetPx=2.12`（右目椭圆中心行偏置预修正）与 `circleAxisRatioMax=0.12`（截面非圆度上限）为本批标定质量代偿值——重标定后 `rowOffsetPx` 归零、`axisMax` 回默认 0.03 并复扫重钉（定档手段：conic_offline `--row-offset scan`，见 §8）。

## 8. 已知限制（先看这个再怀疑工具）

1. **配准大多 hold 多属正常**：本批数据标定极线误差 0.224px 超标，配准门槛受此压制——v3 实测（78 帧）markersValid 中位 5、flow 成链 9/77 帧（11.7%，含首帧）、hold 88.3%、skip 0。**数据质量限制，非工具故障**——换更好标定数据重跑即改善（复跑 `--cycles 0` 出新 report → 重钉 `runtime.check`）。
2. **同帧跑两遍结果可能差 1-2 个点**：GPU 计算顺序固有抖动（激光 3↔4 点）＋标记点链 ccl GPU 标签序/边界拟合抖动（实测 2↔1 个有效圆点且位置可超 0.1mm）。确定性断言按容差口径（最近邻几何比对＋无对应点数≤2）；大幅发散（factory_calib 幽灵线同款）仍报红。
3. **两管共用一张标定表**：第二支管标定到位前，偶数帧与奇数帧同表（已留接入位）。
4. **conic 代偿工作点随标定走**：`rowOffsetPx=2.12`/`circleAxisRatioMax=0.12` 为本标定质量代偿工作点（conic_offline `--row-offset scan` 定档）；重标定后 `rowOffsetPx` 归零、`axisMax` 回 0.03 并复扫（`--cycles 0` 出新 report → 重钉 `runtime.check`）。

## 9. 技术参考（改代码/深究才需要看）

1. **接线序**（`docs/流水线/客户端扫描流水线.md` §一）：`mask_separation → ccl` 分双支——标记点链 8 算子（CPU：image_split → zernike_edge → image_merge → undistort_cpu → ellipse_fit → marker_match → **conic_reconstruct**，圆锥相交重建一步顶替旧 epipolar_intersect/edge_match/point_reconstruct 三步）＋配准 2（optical_flow_fuse/frame_fuse）＋激光链 5 算子（CUDA），汇于 R/T＋标记点 3D＋激光 3D；三级配准降级（光流→relock 宽域重锁→frame_fuse 兜底【matchMode=1 三角形模式＋子集契约：视野捞点快路径/三角形索引慢路径；链内验收门 config runtime.fuseMinMatched=4/fuseMaxRmseMm=2.0，专题见 tests/frame_fuse_opt/】→沿用上帧）。
2. **宽圆斑形态已由算子原生支持（2026-10-04 细激光形态重设计）**：mask_separation 重写为腐蚀级联 3/9/37＋双 top-hat 减法（还原核=杀核+2）＋边缘膨胀 9/7，宽标记点原生进入 `d_markingPointMask`；历史 `swapOutputs` 组合掩膜补丁已删除（设计文档：`docs/plans/2026-10-04-mask-separation-thin-laser-redesign-design.md`）。实测本数据集：真实标记点表观 10-25px（光晕环 60-237px 亦入 marking 靠椭圆质量闸/匹配淘汰）、激光 3D 点 18→415。
3. **右目 P(0,3) 置零**：激光链消费"右目自身列坐标"约定（CMTT 表 uR 域＋Q[14]=1/B 双重实证）；不置零右目 x 偏 -17.1 万像素→v3 全排除。
4. **undistort_cpu 经 `SetRectifyMatrices` 注入温表档** R1/R2/P1/P2/Q（与激光链同源同温，一致性检查实测恒等）。
5. **确定性口径**：GPU 归约/原子顺序＋边界拟合浮点抖动→断言按 |Δ|≤2 容差＋同下标逐位比对；大幅发散（factory_calib 幽灵线同款）仍报红。
6. **07 ScanChains v3 迁移参考**：同帧双链分叉、双管温表注入、右目 P 置零、组合掩膜、温表档一致性检查——均为迁移必读约定。
7. **源码**：`src/` 逐文件说明见 §10；`tests/`（16 gtest 用例，口径见 §6 判卷处）。

## 10. 源码文件说明（src/）

| 文件 | 干什么用的 |
|---|---|
| `main.cpp` | 程序入口。解析命令行参数（`--cycles`、`--step`、`--temp-compare` 等），读取配置文件，然后组织整个回放过程：装载标定 → 逐帧跑流水线 → 把每帧耗时和统计数字写成 CSV → 在屏幕上打印每帧一行结果。跑完后给出汇总（各步骤中位耗时、失败帧数）。 |
| `harness.h` / `harness.cpp` | 核心的"总装车间"。创建全部算子的实例，按流水线文档规定的顺序把它们接起来，逐帧执行整条链（掩膜分离 → 标记点链 → 激光链 → 配准）。它还负责：每一步掐表计时、记下哪步失败、把每步算出的东西写成图片/点表/点云文件。**本工具与原版 scan_chain_replay 的唯一差别就在这里**：steger 出点后多一步平顶去重，走哪条路由 `runtime.dedupPath` 决定——默认 `operator`＝算子内 GPU 去重（steger_fast `dedupMode` 治理，harness 只透传参数）；`cpu`＝外挂 `steger_dedup_merge`（对照 oracle，见下）。 |
| `steger_dedup_merge.h` / `steger_dedup_merge.cpp` | 激光中心点去重——**已并入算子**（steger_fast `dedupMode=1`，2026-10-12）；本模块降级为 CPU 对照 oracle，经 `runtime.dedupPath=cpu` 启用（harness 同时强制算子 dedupMode=0，防双重去重）。steger 算子有时会在几乎同一个位置输出好几个点，这个模块把这样的"挤在一起的一簇点"合并成一个。判断标准：两点沿线方向偏移很小、垂直线方向偏移也很小、且两点处线的走向一致。合并规则：簇里 1 个点直接用、2 个点取平均、3 个及以上取中位。合并完重建算子结果的 CPU 和 GPU 两份数据，并汇报"多少点进、多少点出、合并了多少"。 |
| `stages.h` | 步骤名单表。登记流水线里全部 16 个步骤的名字、每步的直接前置、是否左右目各跑一遍。`--step` 参数就是拿名字在这里查的。 |
| `scr_config.h` / `scr_config.cpp` | 配置文件读取。解析 `config.json` 的三个部分：`paths`（数据/标定/输出路径）、`runtime`（跑几帧、名义温度等）、`operators`（各算子参数）。支持 `{JMW_SOURCE_ROOT}` 占位符（换机器不用改路径）；缺哪项就用默认值，不会抛异常。 |
| `calib_loader.h` / `calib_loader.cpp` | 标定文件装载。解析相机标定 JSON（内参、外参、立体矫正参数），按设定的名义温度从 61 档温度补偿表里挑最接近的一档；再从二进制大文件（约 2.36GB）里只读出对应那一档的激光温表。装完还做一项检查：温表的行距和插值算子要求的行距是否一致，不一致会警告或报错。 |
| `frame_source.h` / `frame_source.cpp` | 帧数据管理。数一数数据目录里有哪些帧、按序号把左右目图片配成对、按帧号奇偶判断是哪支激光管打的（决定用哪份标定参数），并提供"给我第 N 帧的左/右目图片"的读取功能。 |
| `artifacts.h` / `artifacts.cpp` | 结果保存工具箱。一组统一的存盘函数：PNG 图片（含掩膜上色叠加图）、CSV 点表、PLY 三维点云（可拿去 MeshLab/CloudCompare 看）、JSON 文件，以及"首行写表头、之后逐行追加"的 CSV 写法。harness 落盘全走这里。 |

.h 和 .cpp 成对出现时，.h 是说明"提供哪些功能"的头文件，.cpp 是具体实现。
