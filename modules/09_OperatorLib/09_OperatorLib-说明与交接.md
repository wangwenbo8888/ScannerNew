> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# 09_OperatorLib · 算子库

## 1. 目标

- 全部算法算子：标志点链/激光链/预处理/融合/全局优化；OperatorLib.dll＋契约指纹热更；退役算子进 retired/
- 量化锚｜53 个算子目录（在用 44＋闲置/退役候选 9）；每目录自带 tests；指纹不符即拒绝加载
- 注意点｜算子统一三元组契约（Params/Result/Operator＋pImpl＋Stream&）；global_optim 不入三元组为独立组件

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 标志点链：达成｜`core/marker/` 14 算子（image_split→zernike_edge→…→point_reconstruct＋配准 optical_flow_fuse/frame_fuse＋marker_cloud_fuse_cpu）
- 激光链：达成｜`core/laser/` CUDA Stream& 流亲和（steger→undistort_cuda→epipolar_interp→laser_reconstruct）＋`calibration/laser_calib/` 8 件（PJC 主链）＋扫描匹配 laser_match_scan_v3_split 主用
- 预处理：达成｜`core/vision/` 三工序 mask_extract/frame_filter/ccl（CUDA）＋`scanning/preprocess/` mask_separation
- 融合：达成｜`scanning/fusion/` 双轨——laser_cloud_fuse(_cuda) 体素哈希＋laser_cloud_normal(_cuda) 法线（07-E 消费）
- 全局优化：达成｜`scanning/global_optim/` global_ba_cpu＋ba_residuals＋pose_graph_residuals（Ceres GBA＋软先验＋Tukey＋卡方门控）
- OperatorLib.dll＋契约指纹热更：达成｜`api/jmw_op_contract.h`＋`jmw_op_contract_hash.h`（公共头闭包＋工具链串 SHA-256）
- 指纹不符即拒绝加载：达成｜指纹机制在库；「启动校验不一致直接拒绝」为本目录 AGENTS 口径〔待核：现状段无运行时证据〕
- 退役算子进 retired/：未达成｜`retired/` 尚不存在（首次退役算子时建）；闲置/退役候选 9 个去留未裁定（见问题段）
- 53 个算子目录（在用 44＋闲置/退役候选 9）：达成｜每算子目录带 tests/＋两件套；53×两件套铺设见日志 261102〔待核：计数源自日志/问题段，现状段未点名〕
- 每目录自带 tests：部分达成｜绝大多数目录带 tests/＋两件套；epipolar_pair_cpu 明言无测试
- 三元组契约（Params/Result/Operator＋pImpl＋Stream&）：达成｜目录根 README.md＝契约速览（算子规范 v1.9）；Params validate/toJson/fromJson＋Operator 四口＋pImpl 禁拷贝＋QualityFlag 随 Result 不抛业务异常
- global_optim 不入三元组为独立组件：达成｜global_ba_cpu 为准三元组（无 Warmup）；gba_dataset_runner 独立 exe 不入库（BUILD_GLOBAL_OPTIM 门控）

## 3. 问题与方法

- 〔待定〕问题：GBA 真机降级根因未查（迭代=0 初值兜底，疑观测帧/先验点条件不满足）｜方法：真机排查根因｜提出 261102｜解决 —
- 〔待定〕问题：v3 参数无产线出口（urBiasModel 等无源，现走算子默认参）｜方法：接通产线参数源并对拍评估精度｜提出 261102｜解决 —
- 〔待定〕问题：A 模式 `mask_separation` 无激光工况未实测｜方法：真机实测（未经设计验证，见 `docs/流水线/客户端扫描流水线.md` §3.4）｜提出 261102｜解决 —
- 〔待定〕问题：闲置算子 9 个去留未裁定｜方法：endpoint/pose_optimize/virtual_camera/epipolar_intersect/edge_match/point_reconstruct/normal_cuda/epipolar_pair/laser_match_scan v1 逐个裁定｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制（对象级细节精简，证据留真名）｜规范 §2.3
- 261102｜转四段新制：目标待人定稿（含原全局决策注意点）；问题段自下一步+全局挂账迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段按存量文档颗粒度扩充（源：docs/存量档案/模块功能/09＋代码实测校准）｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实；目录级 README 已存在，保留为契约速览）｜基线＝「现状」段｜—
- 261102｜算子目录两件套首批铺设（53 目录 × 说明＋AGENTS；调研发现 7 处闲置/退役候选与 2 处文档滞后已记入各目录说明「下一步」待人工裁定）｜见各算子目录说明｜—
