# TODO — 待办清单

> 生成：2026-08-22（真机联调批次后）。完成项勾选并注明提交号；新增项追加到对应分组尾部。

## 1. 真机联调遗留（2026-08-22 批次，commit 763b7a1 / 23e32bb）

- [ ] **温度量纲未定**：固件裸温度行 ~133（非合理 ℃），暂未入 TempFrame——0x0803/0x0804 温度告警、WarmupSequence 预热空转。等固件方确认量纲/系数后在 `MCUDriver::feedTextLine` 接入
- [ ] **加密狗检测占位**：SelfCheck 项 `license` 恒 true；狗到货后接 USB VID/PID 枚举或厂商 SDK（AppContext.cpp 搜 `license` 定位）
- [ ] **v3 协议对齐**：固件无 v3 实现（无 ACK/CRC/`$` 帧），PC 侧 `串口通讯可靠约定.md` 停留"提案"；与固件方定排期或正式放弃（放弃则 DeviceConfig.protocol 默认 V2 转正、删 v3 死路径）
- [ ] **CH343 慢写根治**：软件 150ms 保活已缓解（帧 55ms 级）；根治需升级驱动（现 wch 2021 v1.5）/串口线换主板后置 USB 口/PCIe 串口卡
- [ ] **关闭扫描仍慢**：停流瞬间 USB 风暴竞速（已加 flushWrites(300) 缓解熄灯帧；体感仍慢再查相机停流本身耗时）
- [ ] **按键链未验**：v2 固件 K 事件上报格式未知，KeyManager 手势/MenuLogic 菜单全链路未在真机验证（协议表无按键上行格式）
- [ ] **相机 N12Z0 后状态**：固件自检模式退出仅以回显为凭，"Self check begins"后是否有结束报文未确认

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

- [ ] **协议对照-实测文档**：v2 固件实测行为（回显一切/裸温度行 7Hz/N12Z1 文本应答/B-L 量程 0-100）与命令表逐条对照落档到 `docs/开发需要的信息/下位机和按键/`（本会话结论未落盘）
- [ ] **环境配置汇总.md 更新**：本机路径已大改（F:/opencv4.13、D:/eigen3-config、F:/osg3.6.5、C:/Qt/Qt5.15.2、Enterprise CRT），文档仍是旧机路径
