> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# 10_Observability · 可观测

## 1. 目标

- 运行监护：7 态状态机、命令门禁、故障链、日志、崩溃捕获、性能监控
- 量化锚｜6 对实现；9 项单测（含并发专项＋崩溃子进程）；7 个状态
- 注意点｜全局唯一状态裁决，一切模式切换都问它；勿在事件回调内调 reportFault（死锁）

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 7 态状态机：达成｜StateMachine S1–S7 表驱动＋CAS 原子转态（19 条合法边；S6 免疫故障、S7 出口唯一 SelfCheckPassed→lastHealthy）
- 命令门禁：达成｜CommandGate submit 三关（注册/门禁/pre）→切态→handler 锁外点火；notifyCompleted 唯一收尾；DefaultCommands.h 7 条
- 故障链：部分达成｜FaultHandler 两口分工（reportFault 完整链：记档→Error→S7＋安全停机→红灯→广播）在库；缺 selfCheckPassed 恢复触发方（生产零调用，S7 无自动恢复出口）＋真机故障注入验证〔待核〕
- 日志：达成｜ObsLogger 双 sink（daily_file 保留 7 份）＋obsExportDiagnosticsPackage 排障包＋jmw_logging.h 宏族（未装配走内建 stdout）
- 崩溃捕获：达成｜CrashHandler crash::install＝SEH filter＋MiniDump＋残留检测（filter 内零堆分配/锁/日志）；启动最先装配（app/main.cpp 定死）
- 性能监控：部分达成｜PerfMonitor 拉取模型＋4 类默认阈值（CPU/内存/磁盘/帧率比）在库；缺处理帧率/丢帧率判定规则（三计数器已接）＋预警级 HealthReport 无 UI 消费
- 全局唯一状态裁决（一切模式切换都问它）：达成｜转态入口收敛铁律——只许命令通道与故障链两个门进；唯一例外＝app 侧 DeviceDisconnected 直连一处
- 勿在事件回调内调 reportFault（死锁）：达成｜EventBus 订阅口仅记档（总线锁内禁 transition/publish）；app 故障桥经 QTimer 延后主线程补 S7
- 6 对实现：达成｜StateMachine/CommandGate/FaultHandler/PerfMonitor/ObsLogger/CrashHandler（15 文件编入 Observability.dll）
- 9 项单测：达成｜tests/ 9 项，含 test_state_machine_concurrent 并发专项（7 态×12 事件＋8 线程×200 次）＋test_crash_handler 崩溃子进程〔待核：通过态未注明〕
- 7 个状态：达成｜S1–S7（canOperate 7 键＋UI 常驻 7 态指示＋设备灯四色回显）

## 3. 问题与方法

- 〔待定〕问题：G03 上报缺失致 0x0808 丢帧检测空转｜方法：随 08 固件对表项（G03 缺失）裁决后复核｜提出 261102｜解决 —
- 〔待定〕问题：故障对表未真机对拍复核｜方法：0x0802/0x0804/0x0808 真机对拍复核｜提出 261102｜解决 —
- 〔待定〕问题：perf 口径（CUDA WDDM）复核沉淀未做｜方法：复核监控口径并沉淀文档｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制（对象级细节精简，证据留真名）｜规范 §2.3
- 261102｜转四段新制：目标待人定稿（含原全局决策注意点）；问题段自下一步+全局挂账迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段按存量文档颗粒度扩充（源：docs/存量档案/模块功能/10＋代码实测校准）｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
