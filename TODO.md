# TODO — 待办清单

> 生成：2026-08-22（真机联调批次后）。完成项勾选并注明提交号；新增项追加到对应分组尾部。

## 1. 真机联调遗留（2026-08-22 批次，commit 763b7a1 / 23e32bb）

- [x] **温度量纲未定**：已随 260831 协议对齐解决（26eb450）——G02 恒四路 0-100℃ 上报，旧裸温度行路径（feedTextLine）已删
- [ ] **加密狗检测占位**：SelfCheck 项 `license` 恒 true；狗到货后接 USB VID/PID 枚举或厂商 SDK（AppContext.cpp 搜 `license` 定位）
- [x] **v3 协议对齐**：已裁决作废（26eb450）——260831 定稿为裸文本无 ACK/CRC，v3/v2 双版本路径已物理删除
- [ ] **CH343 慢写根治**：软件 150ms 保活已缓解（帧 55ms 级）；根治需升级驱动（现 wch 2021 v1.5）/串口线换主板后置 USB 口/PCIe 串口卡
- [ ] **关闭扫描仍慢**：停流瞬间 USB 风暴竞速（已加 flushWrites(300) 缓解熄灯帧；体感仍慢再查相机停流本身耗时）
- [ ] **按键链真机验证**：260831 后 G01 12 手势 MCU 直判、KeyManager 退役、KeySemantics 直收（26eb450）——全链路真机验证待做
- [x] **相机 N12Z0 后状态**：已作废——260831 协议无自检命令（N12Z 删除），该问题不存在

## 2. 工程级待办（AGENTS.md / 模块文档挂账）

- [ ] **10-可观测性故障桥两待办**：① 08 故障 param1 语义对齐 ② 完整链 app 桥（见 docs/模块功能/10 §2.4）
- [ ] **UI 状态图标**：状态机 7 态未上 MainWindow（现仅日志/状态栏文本）
- [ ] **网格四族算子**：07-E 后处理消费方，未建（见 模块功能目录 §09 待建）
- [ ] **05-编辑 / 11-安装部署**：仍是桩模块
- [ ] **BUILD_UI=OFF 恢复评估**：UI 层已实际在构建（scan_demo），根 CMakeLists 开关名存实亡，需清理或转正

## 3. 解锁扫描主链路（建议最高优先）

- [ ] **标定数据产出 calibration.json**：start_scan 一直被门禁拒（缺：相机内参 L/R、外参 R/T、立体温度表 rectify、激光档表 planeMap、meta.imageSize）——用 factory_calib `data_in/`（E 盘版本含真实样本）跑 camera_calib.exe / laser_calib.exe 产出并装载，扫描工作流才能进 S4
- [ ] **data_in 样本回迁**：本仓 factory_calib/data_in 仅 .gitkeep，真实样本在 E:\workfold\factory_calib\data_in（含 camera 棋盘格＋laser pose_00）

## 4. 文档补遗

- [x] **协议对照-实测文档**：v2 固件实测行为对照已无必要——260831 硬切后 v2 行为（回显一切/裸温度行/N12Z1 应答）全部作废；历史行为如需留档从 git 史（≤759f302）取
- [ ] **环境配置汇总.md 更新**：本机路径已大改（F:/opencv4.13、D:/eigen3-config、F:/osg3.6.5、C:/Qt/Qt5.15.2、Enterprise CRT），文档仍是旧机路径
