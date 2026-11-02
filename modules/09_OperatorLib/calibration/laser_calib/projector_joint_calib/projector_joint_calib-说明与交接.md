> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# projector_joint_calib · 投射器联合标定

## 1. 目标

- 多姿态联合解投射器光心＋每线隐式发射曲线：投影机当无旋转针孔虚拟相机，Sampson 距离残差＋手写 LM/IRLS 优化，产出 projectorT＋ImplicitCurve[]（curve_map 的曲线来源）
- 关键参数 maxIterations=100、minPoses=5、curveDegree=3；特有约束 R=I、‖C‖=1、初值须在真值数 mm 内
- 注意：多线版 F8 右目残差仅 baseline>0 启用（改默认须人工批准）；Result 为 move-only；替代旧端点链 endpoint_extract→virtual_camera_pose→pose_optimize

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 多姿态联合解投射器光心＋每线隐式发射曲线：达成｜主类 `ProjectorJointCalib` 把投影机当无旋转针孔虚拟相机，Sampson 距离残差（深度加权 w=Zp/f）＋手写 LM 退火联解 projectorT＋ImplicitCurve[]，约束 R=I、‖C‖=1、初值须在真值数 mm 内，附逐线/逐姿态诊断＋JᵀJ 条件数；仿真精度（σ=0.2mm 噪声、30 seed）tx≈0.17mm/ty≈0.01mm/tz≈1.26mm
- 关键参数与链路定位：达成｜maxIterations=100、minPoses=5、curveDegree=3（SVD 平面内点过滤 0.6mm＋三阶多项式降噪）；在用 07 LaserChain.h/.cpp、CalibSerialize.h/.cpp，curve_map include 本头取 ImplicitCurve；已替代旧端点链 endpoint_extract→virtual_camera_pose→pose_optimize

## 3. 问题与方法

- 〔待定〕问题：激光标定精度待真机验证｜方法：随真机验证演进｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制｜规范 §2.3
- 261102｜转四段新制：目标待人定稿；问题段自下一步迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段按存量参考卡颗粒度扩充（源：docs/存量档案/算子说明文档＋代码实测校准）｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
