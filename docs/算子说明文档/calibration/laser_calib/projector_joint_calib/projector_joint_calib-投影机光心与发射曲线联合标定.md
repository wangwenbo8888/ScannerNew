# 投影机光心与发射曲线联合标定

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 激光标定-14 |
| 中文名称 | 投影机光心与发射曲线联合标定 |
| 英文目录名 | projector_joint_calib |
| 运行平台 | CPU (Eigen 手写LM / 可选 Ceres AutoDiff) |
| 所属流程 | 激光标定流程（独立，暂不接入主流程编排） |
| 精度档次 | ③ 亚像素/浮点类 |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | poses（多姿态3D点云，每姿态一组点+线ID） | vector\<PosePointSet\>（CV_32FC3 + CV_32SC1） |
| **输入②** | f（焦距，像素） | double |
| **输入③** | principalPoint（主点 cx, cy） | cv::Point2d |
| **输入④** | initialT（t初值，机械装配公差） | cv::Vec3d |
| **输出** | ProjectorJointCalibResult 含 projectorT, emissionCurve(coeffs/discriminant/sampsonRms), initialT, jacobianConditionNumber, initialSampsonRms, finalSampsonRms, improvementRatio, denoisedPoints | Vec3d / double[6] / double |

## C. 算法

**核心原理**：将投影机视为针孔虚拟相机（R=I 无旋转），多姿态3D点反投影到虚拟CMOS应聚拢到同一条固定隐式二次曲线 F(u,v)=0。联合优化光心 t(3) + CMOS曲线 C(6) 使 Sampson 残差最小。

**替代旧链路**：旧 endpoint_extract(4-9) → virtual_camera_pose(4-10) → pose_optimize(4-11) 依赖激光线端点（直线交汇求光心），端点不可见时失效。本算子只用激光线可见中段点云（多姿态反投影聚拢），不依赖端点。

**核心流程**：

```
Step 1: 逐姿态 SVD 平面降噪（去深度噪声 Z）
        协方差 SVD → 法向 n（最小特征值）/ uAxis（最大特征值，线主方向）/ vAxis（中特征值，弯曲方向）
        内点过滤 |离面距离| < planeFitInlierThresh(0.6mm) → 投影到平面

Step 1.5: 平面曲线降噪（去横向噪声 X,Y，Step 1 未覆盖）
          平面局部坐标 α=(P−c)·uAxis, β=(P−c)·vAxis
          α 归一化（避免 α³~1e7 导致 bdcSvd 病态）
          curveDegree=3: 3阶多项式拟合 β=aα'³+bα'²+cα'+d
          curveDegree=2: 2阶抛物线 β=aα'²+bα'+c（验证精度差14×，不推荐）
          残差 3σ 过滤 + 重拟合 → 垂直投影 β'=g(α') → 转回3D

Step 2: 坐标归一化（基于 t_init 反投影统计）
        u'=(u-μ_u)/σ_u, v'=(v-μ_v)/σ_v（消除 u²~1e6 导致 JᵀJ 病态）
        曲线初始化 C={0,0,0,0,1,0}（直线 F'=v'，‖C‖=1）

Step 3: 联合优化 t(3)+C(6)=9DOF
        ├─ [useCeres=true] Ceres 后端：
        │    AutoDiff SampsonResidual（雅可比精度1e-15）+ CurveRegularizer（固定正则λ₀）
        │    DENSE_QR trust region（精度与手写LM一致，慢6×）
        │
        └─ [useCeres=false] 手写 LM（默认）：
             残差 r_i = √w_i · F'_i / ‖∇F'_i‖    w_i = Zp_i/f（深度加权）
             正则 r_{N+k} = √λ · C_k              L2正则退火：λ₀=1.0 每iter×0.95→1e-6
             雅可比：数值前向差分（eps=1e-7）
             LM: (JᵀJ+λI)δx = -Jᵀr，C归一化‖C‖=1
             cost平台检测（30-iter窗口降<5%早停，防tz漂移）
             curveDegree=2 时雅可比 B,Cc 列置零 + E=1 归一化（显式2阶模式）

Step 4: 输出与诊断
        反归一化 → 原坐标曲线 emissionCurve.coeffs[6] + discriminant(B²-4AC≤0碗状) + sampsonRms
        退化检测：JᵀJ条件数>1e10 → qualityFlag=Warning（姿态退化，t_z不可信）
            （阈值 1e10：实测正常标定 cond~1e9，正面退化~1e12；旧 1e6 致所有正常标定误报"退化"）
        异常拟合检测：finalSampsonRms>anomalyRmsThreshold(默认0.15) → qualityFlag=Warning（伪极小值/错解）
            （cond 查不出伪极小值——其 cond 反而更低~1e7；rms 是有效探测器：σ=0.2 正常~0.045，陷阱~0.48）
```

**关键数学模型**：

| 模型 | 公式 | 说明 |
|------|------|------|
| CMOS反投影（R=I） | u = f·(X−t_x)/(Z−t_z) + c_x | 投影机系深度 Zp = Z−t_z |
| 隐式二次曲线 | F(u,v) = Au² + Buv + Cv² + Du + Ev + F₀ = 0 | ‖[A..F₀]‖=1 消除尺度冗余 |
| Sampson距离 | d = F / √(F_u² + F_v²) | 一阶几何近似，平缓段也有梯度 |
| 深度加权 | w = Zp / f | 经验值，比严格1/σ²更稳定（实验验证） |
| L2正则退火 | 代价 += λ(t)·‖[A,B,C]‖², λ₀=1.0→1e-6 | 初期压弯曲强迫先调t，解t_z-弯曲耦合 |
| cost平台检测 | 30-iter窗口 Σcost 降<5% → break | 防t_z真值小时末期漂移 |

**关键第三方函数**：

| 函数 | 用途 |
|------|------|
| Eigen::SelfAdjointEigenSolver | SVD平面降噪（协方差特征分解）+ 退化检测（JᵀJ条件数） |
| Eigen::bdcSvd | Step 1.5 曲线拟合（最小二乘） |
| Eigen::LDLT | LM正规方程 (JᵀJ+λI)δx=−Jᵀr |
| [可选] ceres::AutoDiffCostFunction | Ceres后端（AutoDiff雅可比精度1e-15，需BUILD_CERES） |

## D. 依赖

**上下游算子**：

```
laser_reconstruct(4-8) → 本算子（独立，暂不接入主流程）
```

本算子替代旧的 endpoint_extract(4-9) → virtual_camera_pose(4-10) → pose_optimize(4-11) 链路。
旧链路依赖激光线端点（直线交汇求光心），端点不可见时失效。
本算子只用激光线可见中段点云（多姿态反投影聚拢），不依赖端点。

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| `common/calib_types.h` | QualityFlag 等共享类型 |
| `common/calib_logging.h` | 统一日志宏 |
| `common/calib_warmup_config.h` | WarmupConfig 结构体 |
| `common/version.h` | OperatorInfo / SCANNER_VERSION |

## D2. 衔接

**上游→本算子**（编排层负责聚合多姿态）：

| 来源 | 传递方式 | 说明 |
|------|---------|---------|
| laser_reconstruct(4-8) | CPU（编排层download+聚合） | 每姿态3D点 → PosePointSet → poses |
| 双目标定(3-4) | 值传递 | f, principalPoint |
| 机械装配 | 值传递 | initialT（光心初值，如(80,3,3)±1~3mm） |

**本算子→下游**（后期迁移时）：

| 输出字段 | 传递给 | 说明 |
|---------|--------|------|
| projectorT | plane_map(4-12) | 光心 t（替代旧 virtualT） |
| emissionCurve | plane_map / laser_match_scan | CMOS曲线（替代旧 LaserLineCurve） |

> 当前独立存在，下游仍用旧 pose_optimize。后期迁移时改下游消费方。

## E. 架构

**文件结构**：

```
projector_joint_calib/
├── projector_joint_calib.h          # 公开接口（Params/Input/Result/Operator 三元组）
├── projector_joint_calib.cpp        # 实现（Step 1-4 全部 + Ceres后端条件编译 #if BUILD_CERES）
└── tests/
    └── test_projector_joint_calib.cpp   # 15个测试（含精度对比/退化/截断/带宽法/质量标记验证）
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | ProjectorJointCalib（无状态，无pImpl——CPU算子） |
| 核心方法 | Execute(const ProjectorJointCalibInput&) |
| 预热方法 | Warmup()（空实现，CPU标定算子统一接口形状） |
| 参数更新 | SetParams() / GetParams() |
| 参数结构体 | ProjectorJointCalibParams |
| 输入结构体 | ProjectorJointCalibInput |
| 结果结构体 | ProjectorJointCalibResult（move-only） |
| 曲线结构体 | ImplicitCurve（coeffs[6], discriminant, sampsonRms） |
| 日志标签 | "14-ProjectorJointCalib" |

**状态模型**（算子规范 §4）：

| 项目 | 说明 |
|------|------|
| 状态类别 | 无状态 |
| 说明 | 多姿态点云/焦距/初值按调用传入；Execute 一次性求解 |
| 并发策略 | 每实例非线程安全（§1.4），多实例并行各自独占 |

**错误模型**（算子规范 §5.1）：

单一错误模型 `bool success + QualityFlag qualityFlag + std::string message`。

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| Eigen | >= 3.4 | 线性代数（SVD/LDLT/特征分解/LM） |
| OpenCV | >= 4.x | 矩阵数据结构（Vec3d/Point2d） |
| nlohmann_json | — | Params 序列化 |
| Ceres（可选） | 2.2.0 | useCeres=true 时（BUILD_GLOBAL_OPTIM=ON → BUILD_CERES=1） |

## F. 参数

### 一期参数（已实现，实验验证）

| 参数名 | 类型 | 默认值 | 范围 | 说明 | 调参历史 |
|--------|------|--------|------|------|---------|
| maxIterations | int | 100 | >0 | LM/Ceres最大迭代 | 测试用200 |
| convergenceThreshold | double | 1e-8 | >0 | 步长收敛 | — |
| minPoses | int | 5 | >=3 | 最少姿态数 | — |
| minPointsPerPose | int | 50 | >=10 | 每姿态最少点 | — |
| planeFitInlierThresh | double | **0.6** | >0 | 平面降噪内点阈值（3σ for σ=0.2mm） | **0.05→0.6 重大修复**（planePts 20%→99.5%，tz 2.37→0.44mm） |
| lambda0 | double | 1.0 | >0 | L2正则初值（强正则压弯曲） | — |
| lambdaDecay | double | 0.95 | (0,1) | 正则每iter衰减率 | — |
| curveDegree | int | **3** | 2或3 | 曲线阶数（3=三阶+隐式6参数） | 2阶验证精度差14×（圆锥曲线段需3阶） |
| useCeres | bool | **false** | — | 优化后端（false=手写LM, true=Ceres） | Ceres精度一致但慢6×，作交叉验证用 |
| anomalyRmsThreshold | double | **0.15** | >0 | 异常拟合检测阈值（finalSampsonRms 超此值判 Warning） | σ=0.2 正常 rms~0.045，伪极小值陷阱~0.48，取 0.15 留 3 倍余量 |

### 二期预留（Params有字段，未实现）

| 参数名 | 类型 | 默认值 | 预期作用 |
|--------|------|--------|---------|
| huberToCauchyThresh | double | 1.0 | 鲁棒核退火 τ₁（Huber→Cauchy） |
| cauchyToL2Thresh | double | 0.3 | 鲁棒核退火 τ₂（Cauchy→L2） |
| topologyEpsilon | double | 1e-3 | 碗状拓扑软惩罚 ε |

## G. 约束

| 约束类型 | 指标 | 选择依据 |
|---------|------|---------|
| 几何假设 | R=I（投影机与左相机无旋转，绝对平行装配） | txt §一.2 |
| 优化自由度 | t(3) + C(6) = 9 DOF（R/K 固定） | — |
| 曲线约束 | ‖C‖=1（消除尺度冗余） | — |
| CMOS曲线形状 | 隐式二次曲线（6参数），discriminant B²−4AC≤0 碗状 | txt §三.2 |
| 深度加权 | w=Zp/f（经验值） | 比严格1/σ²更稳定（w=Zp²场景依赖、w=1近点主导，均实验淘汰） |
| 平面阈值 | 0.6mm（3σ for σ=0.2mm） | 旧0.05mm仅保留20%点（隐藏bug） |
| 曲线降噪阶数 | 3阶多项式（非2阶/隐式圆锥曲线） | 2阶模型误差1.45mm，隐式圆锥曲线噪声崩 |
| 优化后端 | 手写LM（非Ceres） | Ceres交叉验证一致（0.443 vs 0.440mm），慢6×无必要 |
| t_z 可辨识度 | 残差谷在 t_z-弯曲方向平坦（rms 对 t_z 不敏感） | t_z 跨 seed 方差是物理地板 k·σ²；优化器微调（Marquardt 等）实测无效，降 t_z 只能降上游 σ |
| 吸引盆 | 初值需在真值数 mm 内 | 落入伪极小值会静默错解（t_z 偏~15mm）；已由 rms 异常检测（anomalyRmsThreshold）兜底 |
| 编译选项 | MSVC 需 /bigobj；Ceres 后端需 BUILD_GLOBAL_OPTIM=ON | — |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | improvementRatio ≤ 0.5 且 条件数 < 1e10 且 rms < anomalyRmsThreshold |
| Degraded | 降级 | improvementRatio 0.5~0.9 |
| Warning | 警告 | improvementRatio > 0.9 或 条件数 > 1e10（姿态退化）或 rms ≥ anomalyRmsThreshold（异常拟合/伪极小值） |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败 (validate) | 抛出 std::invalid_argument |
| 输入为空 / f≤0 | 返回 success=false |
| 有效姿态不足 (< minPoses) | 返回 success=false |
| 降噪后无内点 | 返回 success=false |
| OpenCV / std 异常 | 捕获并返回 success=false |

## H. 风险

| 严重程度 | 风险描述 | 影响 | 实验数据 |
|:--------:|---------|------|---------|
| 🟡 中 | t_z 信号弱（t_true.z≈3mm 光心几乎在相机平面） | t_z 精度高度 seed 依赖 | 30 seed@σ=0.2 均值 1.26mm，范围 0.09~3.64mm |
| 🟡 中 | 激光线截断（两端不可见）增大 t_z 方差 | 完整→截断均值 0.90→1.54mm（1.7×） | tx/ty 不受影响 |
| 🟡 中 | t_z ≈ k·σ²（30-seed k≈31，5-seed k≈22） | t_z 是噪声二阶效应，非算法 bug | 残差谷在 t_z-弯曲方向平坦，单次解确定性但受噪声偏置 |
| 🟡 中 | **伪极小值陷阱**：特定初值方向(~4mm)使优化落入 t_z 偏~15mm 的错解 | 静默错解风险（success=true） | 已由 rms 异常检测兜底（陷阱 rms~0.48 >> 正常~0.045）；cond 查不出（陷阱 cond~1e7 反而更低） |
| 🟡 中 | **吸引盆小**：初值需在真值数 mm 内 | 对机械装配初值精度敏感 | 干净场景 +5mm 方向扰动即落入不同盆地（多初值一致性校验因此误报率 100%，不可用） |
| 🟢 低 | 数值雅可比精度 1e-7（vs 解析 1e-15） | LM鲁棒，影响可忽略 | Ceres后端tz=0.443 vs 手写=0.440 |
| 🟢 低 | **阻尼/正则微调不降 t_z 均值** | 避免无效优化尝试 | 30-seed 实测 Marquardt 缩放/costS 平台对 t_z 均值无系统改善（full Marquardt 16好14坏，均值 −1.4% 噪声内） |
| 🟢 低 | 仅优化 t，未优化 R/K | R=I假设 | 若实际有旋转需后续扩展 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |

**精度特征（真实硬件参数 t_true=(80,3,3), f=1500px）**：

| 噪声 σ | tx 均值 | ty 均值 | tz 均值 | tz 范围 | 说明 |
|---------|---------|---------|---------|---------|------|
| 0（无噪声） | 0.01mm | 0.001mm | 0.07mm | — | 算法精度上限（模型误差） |
| 0.05mm | — | — | 0.03mm | — | 低噪声场景（5 seed） |
| 0.10mm | — | — | 0.13mm | — | 中等噪声（5 seed） |
| 0.15mm | — | — | 0.24mm | — | （5 seed） |
| 0.20mm（产线） | **0.17mm** | **0.01mm** | **1.26mm** | 0.09~3.64mm | **30 seed**（中位数 1.02mm；旧 5-seed 子集偏乐观报 0.90mm） |
| 0.20mm+截断 | 0.16mm | 0.01mm | **1.54mm** | 0.06~3.34mm | 激光线两端缺5~20%（5 seed） |

> - t_z ≈ k × σ²（30-seed @ σ=0.2 实测 k≈31；5-seed 子集 k≈22，seed 数/选取影响大）。降 σ 一半 → t_z 降 4 倍（平方级杠杆）。
> - 产线 t_z < 0.05mm 需 σ < 0.05mm（上游双目重建精度）。
> - tx/ty 稳定可靠（<0.2mm，不受噪声/截断影响）。
> - 增加姿态数不降 k（近点主导 JᵀJ 的 t_z 对角，远点贡献微）。
> - t_z 跨 seed 方差源于残差谷在 t_z-弯曲方向平坦（单次解确定性，但受噪声样本偏置），属数据可辨识度极限，非优化器问题（见 H/G）。

**关键实验验证**：

| 实验 | 结论 |
|------|------|
| Ceres 交叉验证 | 手写LM 正确（tz 0.440 vs Ceres 0.443mm） |
| w=Zp/f vs Zp²/f vs 1 | Zp/f 最稳定（Zp²场景依赖，1近点σ依赖） |
| 3阶 vs 2阶多项式 | 3阶优14×（圆锥曲线段需3阶近似） |
| 姿态数 8→25 | k不降（t_z不随N改善） |
| 截断5~20% | tz方差增1.7×（非系统性偏移），tx/ty不受影响 |
| 带宽法 tz 搜索 | v-bin 带宽对 t_z 不敏感（t_z 错位在 u 方向） |
| 平面阈值0.05→0.6 | 隐藏bug修复（planePts 20%→99.5%，tz最大单项改善） |
| **Marquardt/costS 优化微调（30 seed）** | 对 t_z 均值无可靠改善（full Marquardt 16好14坏，均值 −1.4% 噪声内）；t-only 30/30 与 baseline 完全一致 → 证实 t_z 方差是数据可辨识度问题，非优化器问题 |
| **cond 阈值标定** | 正常标定 cond~1e9，正面退化~1e12，几何均值~6e10；旧阈值 1e6 致所有正常标定误报"退化" → 改为 1e10 |
| **rms 伪极小值探测** | 陷阱 t_z 偏~15mm 但 cond~1e7（查不出），rms~0.48（正常~0.045，10× 差距）→ rms 是有效探测器，已落地为 anomalyRmsThreshold |
| **多初值一致性校验** | 不可用：抓陷阱 4/4 但干净场景误报 6/6（吸引盆小，+5mm 方向扰动即换盆）→ 撤销该思路 |
