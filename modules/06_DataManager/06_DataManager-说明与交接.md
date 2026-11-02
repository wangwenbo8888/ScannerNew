> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# 06_DataManager · 数据管理

## 1. 目标

- 数据底座：标定仓库/点云仓库/帧环形缓冲/帧增强（按帧温查表）/会话记账/终局产物落盘；采图与计算解耦（量化锚：30 文件＝10 对实现＋10 纯头、5 项单测，261102 实测）
- 注意点：缺表拒绝开扫；SlotRing 单生产者铁约（仅采集回调单线程写、槽位构造期定死无 resize）

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 标定仓库：达成｜CalibrationRepository（JSON 单文件＝07 序列化载荷＋外层 meta，mutex 全量护，解析→校验→临时文件＋原子改名；默认 config/calibration.json＋合并工厂档）＋CalibSessionData（load 三键：initialParams 透传/targets 每条 16 数/boardPoints，后两键缺或空数组即拒绝）
- 缺表拒绝开扫：达成｜两表全空 assemble 必 fail；readyForScan 查缺样清单（缺哪报哪），stereo()/stereoTempTable()/planeMapTiers() 类型化取数
- 点云仓库：达成｜PointCloudBuffer 双实现 IPointCloudSink＋ICloudWarehouse，shared_mutex（融合写/渲染并发读）；beginCloudSession 折叠既有为基线＋pushSessionCloud 快照替换单调不减；标志点 setMarkers/snapshotMarkers 独立锁＋版本号（globalId 只走内存，落盘只有坐标）；exportCloud/exportMarkers 走 file_io 按扩展名分派（file_io 兼收 ply/pcd/xyz/txt＋STL/OBJ 网格＋标志点 JSON，PCD 头缺陷锁定不修）；旧写口 pushPointCloud/replacePointCloud 待正式口稳定后退役〔待核〕
- 帧环形缓冲：达成｜SlotRing<T> 原子槽位环（Overwrite 扫描环 16 槽满则覆盖最旧永不阻塞／Backpressure 姿态环 8 槽阻塞等 done() 腾位）＋FrameBuffer（IFrameSink 保最新帧，默认 60 槽 DropOldest）＋RingBuffer（溢出三策略 DropOldest/DropNewest/Block，pop 带超时）；姿态环生产写侧未接（01 侧 TODO，真机联调补）〔待核〕
- SlotRing 单生产者铁约：达成｜单 mutex＋双条件变量（写侧等腾位/读侧等新帧），仅采集回调单线程写、槽位构造期定死无 resize；读法契约先 waitFor(frameId) 再 read()（近似语义）
- 帧增强（按帧温查表）：达成｜FrameEnricher 无状态纯函数：帧温 0.2℃ 阶梯就近选档（并列取低温档）填 CalibSnapshot 入 EnhancedFrame（灰度 host 副本＋T/V 判定；档数据形态 TempTableTypes——立体侧 Matx、激光侧仅档温索引归 07/09 显存）；选档失败丢帧不写环＋droppedFrames 计数（首丢告警），越界钳制首末档＋warning 仍入环
- 会话记账：达成｜ScanSessionData assemble 装两表＋pushFrame 帧唯一入口（查表→写 16 槽 Overwrite 环）；CycleUnit 周期打包（marker 左右＋N×2 激光管帧偶左奇右＋周期温度）
- 终局产物落盘：部分达成｜WorkflowArtifactStore L4 字节仓就绪（put/get/list/remove＋key 消毒映射文件名、put 临时文件再改名、版本头 [8B 魔术字][4B 版本][4B 长度] 不匹配返失败不崩），02 装配期探查 markers_gba（软先验＋seed）、终局遍三产物落 artifacts/；缺 WorkflowArtifactStore 用例（test_file_io 仅死 include）＋GB 级激光云不入 L4 仓只存索引
- 采图与计算解耦：达成｜三 Sink 契约 IFrameSink/IPointCloudSink/IDeviceStateSink（依赖倒置，08 不反依赖实现；铁律「06 不链 08」，测试 exe 叶子例外）＋IWorkflow（initialize/start/pause/resume/stop，01/02/04 实现）；配套 DeviceStateCache（按 deviceId 只存最新，08 写/UI 轮询读）、ParameterManager（variant 四类型＋类型化 getter＋相机/扫描预设组，app 持有注入）；点云 Sink 现仅测试路径用〔待核〕
- 量化锚 30 文件＋5 项单测：达成｜261102 实测 30 文件＝10 对编入 DataManager.dll＋10 纯头（接口 5 件不入库源）；tests 5 项 38 用例＝calib_repository 17/cloud_warehouse 6/scan_session 4/calib_session 3/file_io 8

## 3. 问题与方法

- 〔待定〕问题：droppedFrames 与 10 故障桥对账口径复核｜方法：对齐两侧丢帧/故障计数定义，统一归属与报表口径｜提出 261102｜解决 —
- 〔待定〕问题：L4 落盘与 04 免文件中转链路验收｜方法：验收 artifacts/ 三产物落盘＋04 STL 导出免文件中转全链｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制（对象级细节精简，证据留真名）｜规范 §2.3
- 261102｜转四段新制：目标待人定稿；问题段自下一步+全局挂账迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段按存量文档颗粒度扩充（源：docs/存量档案/模块功能/06＋代码实测校准）｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
