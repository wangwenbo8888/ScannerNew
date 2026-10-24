# 产线集成差异台账 · ops ↔ modules/OperatorLib（Phase 0 冻结基线）

> 生成：2026-10-24（MD5 逐文件对照，非 tests、非 global_optim）
> 统计：**同 34 ｜ 异 49 ｜ ops 独有 45 ｜ 主库独有 52**（计 180 文件）
> 处置代号：**A**=ops 版整体替换主库（三方合并保主库修复）｜**B**=ops 新增目录直接入主库｜**C**=主库保留不动（沙盒不需要）｜**D**=随五链 Phase 2 定去留

## 一、异（49）——Phase 1 三方合并清单（重灾区）

```
异	calibration\camera\stereo_rectify_temp_table\stereo_rectify_temp_table_cpu.cpp
异	calibration\camera\stereo_rectify_temp_table\stereo_rectify_temp_table_cpu.h
异	calibration\laser_calib\laser_label\laser_label_cuda_impl.cu
异	calibration\laser_calib\laser_label\laser_label_cuda_pimpl.h
异	calibration\laser_calib\laser_label\laser_label_cuda.h
异	calibration\laser_calib\laser_match\laser_match_cuda_impl.cu
异	calibration\laser_calib\laser_match\laser_match_cuda_pimpl.h
异	calibration\laser_calib\projector_joint_calib\projector_joint_calib.cpp
异	calibration\laser_calib\projector_joint_calib\projector_joint_calib.h
异	calibration\temp\extrinsic_compensate\extrinsic_compensate_cpu.cpp
异	calibration\temp\extrinsic_compensate\extrinsic_compensate_cpu.h
异	calibration\temp\intrinsic_compensate\intrinsic_compensate_cpu.cpp
异	calibration\temp\intrinsic_compensate\intrinsic_compensate_cpu.h
异	calibration\temp\laser_extrinsic_compensate\laser_extrinsic_compensate_cpu.cpp
异	calibration\temp\laser_extrinsic_compensate\laser_extrinsic_compensate_cpu.h
异	core\common\calib_result_types.h
异	core\common\zitai_result_types.h
异	core\laser\laser_reconstruct\laser_reconstruct_cuda_impl.cu
异	core\laser\laser_reconstruct\laser_reconstruct_cuda_pimpl.h
异	core\laser\undistort_cuda\undistort_points_cuda_impl.cu
异	core\marker\edge_match\edge_match_cpu.cpp
异	core\marker\edge_match\edge_match_cpu.h
异	core\marker\ellipse_fit\ellipse_fit_cpu.cpp
异	core\marker\ellipse_fit\ellipse_fit_cpu.h
异	core\marker\frame_fuse\frame_fuse_cpu.cpp
异	core\marker\frame_fuse\frame_fuse_cpu.h
异	core\marker\marker_match\marker_match_cpu.cpp
异	core\marker\optical_flow_fuse\marker_optical_flow_fuse_cpu.cpp
异	core\marker\optical_flow_fuse\marker_optical_flow_fuse_cpu.h
异	core\marker\point_reconstruct\point_reconstruct_cpu.cpp
异	core\marker\point_reconstruct\point_reconstruct_cpu.h
异	core\marker\zernike_edge\zernike_edge_cpu.cpp
异	core\scheduler\prev_frame_state.h
异	core\vision\ccl\region_analyze_cuda_impl.cu
异	core\vision\ccl\region_analyze_cuda_pimpl.h
异	core\vision\ccl\region_analyze_cuda.cpp
异	core\vision\ccl\region_analyze_cuda.h
异	core\vision\mask_extract\mask_extract_cuda_impl.cu
异	core\vision\mask_extract\mask_extract_cuda_pimpl.h
异	core\vision\mask_extract\mask_extract_cuda.h
异	scanning\fusion\laser_cloud_fuse_cuda\laser_cloud_fuse_cuda_impl.cu
异	scanning\fusion\laser_cloud_fuse_cuda\laser_cloud_fuse_cuda_pimpl.h
异	scanning\fusion\laser_cloud_fuse_cuda\laser_cloud_fuse_cuda.cpp
异	scanning\fusion\laser_cloud_fuse_cuda\laser_cloud_fuse_cuda.h
异	scanning\preprocess\mask_separation\example_usage.cpp
异	scanning\preprocess\mask_separation\laser_markingpoint_mask_separation_cuda_impl.cu
异	scanning\preprocess\mask_separation\laser_markingpoint_mask_separation_cuda_pimpl.h
异	scanning\preprocess\mask_separation\laser_markingpoint_mask_separation_cuda.cpp
异	scanning\preprocess\mask_separation\laser_markingpoint_mask_separation_cuda.h
```
注：undistort/ccl/mask_extract/laser_label/laser_match 等含主库 261017-18 两周修复（流序/NonBlocking/pinned 回读）——**合并时主库修复优先保留**，ops 性能改动叠加其上。

## 二、ops 独有（45）——Phase 1 新增（B）或五链配套（D）

```
ops独有	calibration\laser_calib\curve_map\curve_map.cpp
ops独有	calibration\laser_calib\curve_map\curve_map.h
ops独有	calibration\laser_calib\curve_map_temp_table\cmtt_container.cpp
ops独有	calibration\laser_calib\curve_map_temp_table\cmtt_container.h
ops独有	calibration\laser_calib\curve_map_temp_table\curve_map_temp_table.cpp
ops独有	calibration\laser_calib\curve_map_temp_table\curve_map_temp_table.h
ops独有	calibration\laser_calib\curve_map_temp_table\temp_param_interpolator.cpp
ops独有	calibration\laser_calib\curve_map_temp_table\temp_param_interpolator.h
ops独有	core\common\testutil\marker_synth.cpp
ops独有	core\common\testutil\marker_synth.h
ops独有	core\common\epipolar_types.h
ops独有	core\common\steger_types.h
ops独有	core\laser\epipolar_interp_dual\epipolar_interp_dual_cuda_impl.cu
ops独有	core\laser\epipolar_interp_dual\epipolar_interp_dual_cuda_pimpl.h
ops独有	core\laser\epipolar_interp_dual\epipolar_interp_dual_cuda.cpp
ops独有	core\laser\epipolar_interp_dual\epipolar_interp_dual_cuda.h
ops独有	core\laser\flat_center\flat_center_impl.cu
ops独有	core\laser\flat_center\flat_center_pimpl.h
ops独有	core\laser\flat_center\flat_center.cpp
ops独有	core\laser\flat_center\flat_center.h
ops独有	core\laser\laser_reconstruct\laser_reconstruct_cpu.cpp
ops独有	core\laser\laser_reconstruct\laser_reconstruct_cpu.h
ops独有	core\laser\steger_fast\steger_fast_impl.cu
ops独有	core\laser\steger_fast\steger_fast_pimpl.h
ops独有	core\laser\steger_fast\steger_fast.cpp
ops独有	core\laser\steger_fast\steger_fast.h
ops独有	core\marker\conic_reconstruct\conic_reconstruct_cpu.cpp
ops独有	core\marker\conic_reconstruct\conic_reconstruct_cpu.h
ops独有	core\marker\ellipse_fit\ellipse_fit_simd.cpp
ops独有	core\marker\ellipse_fit\ellipse_fit_simd.h
ops独有	core\marker\ellipse_fit\ellipse_fit_vendor.cpp
ops独有	core\marker\ellipse_fit\ellipse_fit_vendor.h
ops独有	scanning\laser\laser_match_scan_v3\cmtt_table_loader.cpp
ops独有	scanning\laser\laser_match_scan_v3\cmtt_table_loader.h
ops独有	scanning\laser\laser_match_scan_v3\laser_match_scan_v3_cuda_impl.cu
ops独有	scanning\laser\laser_match_scan_v3\laser_match_scan_v3_cuda_pimpl.h
ops独有	scanning\laser\laser_match_scan_v3\laser_match_scan_v3_cuda.cpp
ops独有	scanning\laser\laser_match_scan_v3\laser_match_scan_v3_cuda.h
ops独有	scanning\laser\laser_match_scan_v3\laser_match_scan_v3_split_front.cu
ops独有	scanning\laser\laser_match_scan_v3_cpu\laser_match_scan_v3_cpu.cpp
ops独有	scanning\laser\laser_match_scan_v3_cpu\laser_match_scan_v3_cpu.h
ops独有	scanning\laser\laser_match_scan_v3_split\laser_match_scan_v3_split_cpu.cpp
ops独有	scanning\laser\laser_match_scan_v3_split\laser_match_scan_v3_split_gpu.cpp
ops独有	scanning\laser\laser_match_scan_v3_split\laser_match_scan_v3_split.h
ops独有	scanning\preprocess\mask_separation\mask_sep_cli.cpp
```

## 三、主库独有（52）——默认 C 保留，标注例外

```
calibration\camera\extrinsic_calib\extrinsic_calib_cpu.cpp
calibration\camera\extrinsic_calib\extrinsic_calib_cpu.h
calibration\camera\intrinsic_calib\intrinsic_calib_cpu.cpp
calibration\camera\intrinsic_calib\intrinsic_calib_cpu.h
calibration\camera\inverse_distort\inverse_distort_cpu.cpp
calibration\camera\inverse_distort\inverse_distort_cpu.h
calibration\camera\stereo_rectify\stereo_rectify_cpu.cpp
calibration\camera\stereo_rectify\stereo_rectify_cpu.h
calibration\laser_calib\endpoint_extract\endpoint_extract_cuda_impl.cu
calibration\laser_calib\endpoint_extract\endpoint_extract_cuda_pimpl.h
calibration\laser_calib\endpoint_extract\endpoint_extract_cuda.cpp
calibration\laser_calib\endpoint_extract\endpoint_extract_cuda.h
calibration\laser_calib\plane_map\plane_map_cuda_impl.cu
calibration\laser_calib\plane_map\plane_map_cuda_pimpl.h
calibration\laser_calib\plane_map\plane_map_cuda.cpp
calibration\laser_calib\plane_map\plane_map_cuda.h
calibration\laser_calib\plane_map\virtual_pixel_gen.cu
calibration\laser_calib\plane_map\virtual_pixel_gen.h
calibration\laser_calib\plane_map_temp_table\plane_map_temp_table.cpp
calibration\laser_calib\plane_map_temp_table\plane_map_temp_table.h
calibration\laser_calib\pose_optimize\pose_optimize_cuda_impl.cu
calibration\laser_calib\pose_optimize\pose_optimize_cuda_pimpl.h
calibration\laser_calib\pose_optimize\pose_optimize_cuda.cpp
calibration\laser_calib\pose_optimize\pose_optimize_cuda.h
calibration\laser_calib\virtual_camera_pose\virtual_camera_pose_cuda_impl.cu
calibration\laser_calib\virtual_camera_pose\virtual_camera_pose_cuda_pimpl.h
calibration\laser_calib\virtual_camera_pose\virtual_camera_pose_cuda.cpp
calibration\laser_calib\virtual_camera_pose\virtual_camera_pose_cuda.h
calibration\posture\pose_estimate\pose_estimate_cpu.cpp
calibration\posture\pose_estimate\pose_estimate_cpu.h
core\laser\epipolar_interp\epipolar_interp_cuda_impl.cu
core\laser\epipolar_interp\epipolar_interp_cuda_pimpl.h
core\laser\epipolar_interp\epipolar_interp_cuda.cpp
core\laser\epipolar_interp\epipolar_interp_cuda.h
core\laser\steger\steger_extract_cuda_impl.cu
core\laser\steger\steger_extract_cuda_pimpl.h
core\laser\steger\steger_extract_cuda.cpp
core\laser\steger\steger_extract_cuda.h
scanning\fusion\laser_cloud_fuse\laser_cloud_fuse_cpu.cpp
scanning\fusion\laser_cloud_fuse\laser_cloud_fuse_cpu.h
scanning\fusion\laser_cloud_normal\laser_cloud_normal_cpu.cpp
scanning\fusion\laser_cloud_normal\laser_cloud_normal_cpu.h
scanning\fusion\laser_cloud_normal_cuda\laser_cloud_normal_cuda_impl.cu
scanning\fusion\laser_cloud_normal_cuda\laser_cloud_normal_cuda_pimpl.h
scanning\fusion\laser_cloud_normal_cuda\laser_cloud_normal_cuda.cpp
scanning\fusion\laser_cloud_normal_cuda\laser_cloud_normal_cuda.h
scanning\laser\epipolar_pair_cpu\epipolar_pair_cpu.cpp
scanning\laser\epipolar_pair_cpu\epipolar_pair_cpu.h
scanning\laser\laser_match_scan\laser_match_scan_cuda_impl.cu
scanning\laser\laser_match_scan\laser_match_scan_cuda_pimpl.h
scanning\laser\laser_match_scan\laser_match_scan_cuda.cpp
scanning\laser\laser_match_scan\laser_match_scan_cuda.h
```
例外：plane_map 系 8 文件（D2 curve_map 切换后退役）；steger/epipolar_interp 旧版（D3 保留一周期）；laser_match_scan v1（同 D3）；fusion CPU/normal 件（D3）。

## 四、同（34）——零动作
直接以现主库文件为准（含主库后续修复），ops 不再回拷。

## 五、处置汇总（Phase 1 分批依据）

| 批 | 内容 | 文件数 | 验收 |
|---|---|---|---|
| 1-B | core/laser 三新件＋steger_fast/flat_center/epipolar_interp_dual/conic_reconstruct/ellipse SIMD | ~20 | 单测+回放 |
| 2-B | laser_match_v3 三变体＋cmtt_table_loader＋curve_map 系 8 文件 | ~17 | 单测+回放 |
| 3-A | mask_separation/fuse_cuda/zernike/ellipse 三方合并（保主库修复） | ~12 | 0 diffs |
| 4-A | 其余异文件逐个合并（marker/vision/common/scheduler） | ~37 | 0 diffs |
| 5-D | prev_frame_state/调度类型随 Phase 2 五链 | ~5 | 五链验收 |

> 详单逐文件处置表（Phase 1 开工时按批展开到行级，含每文件合并策略：取 ops/取主库/手工融合）。
> 沙盒性能证据与回滚口：`modules/09_operatorlib/TEST/speed_optimize/归档/记录文档/`（43 份）。

