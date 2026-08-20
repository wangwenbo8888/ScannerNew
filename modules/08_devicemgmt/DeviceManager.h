#pragma once
// ============================================================================
// DeviceManager.h — 设备管理门面 + 逻辑线程（D-T12b；设计方案 §4 五增量）
//
// 组合根：MCUDriver/KeyManager/KeySemantics/MenuLogic/ParamStore/WarmupSequence/
// ModeController 全 unique_ptr 内部持有（铁规：不漏零件指针/类型——对外只出值
// 类型与薄转发；MCUDriver/KeySemantics 仅前向声明）。门禁回调注入不反链 07/10
// （GateQuery 只进不出）。
//
// 线程纪律（Critical #1 修正，设计 §6.1「切模式经队列转逻辑线程执行」）：
//   - 无锁零件（CommandChannel 表/ParamStore map/MenuLogic/Warmup/KeyManager）
//     单一属主=逻辑线程；门面一切变异入口经 post() 任务队列编队（mutex+deque，
//     容量 64 满丢新+warn），logicTick 开头排空执行；manualTick 模式下测试
//     调 logicTick 即驱动。
//   - 同步返回值口径：门禁检查（ModeController::request——atomic 读+gate 回调，
//     gate 回调的线程安全性由 app 保证）在调用方线程同步做，拒→不入队同步
//     返回 fail；过→命令组编队执行，返回 request 结果（组成败异步经
//     Fault/StateChanged 观测）。相机三口的返回值=前置检查（无相机同步 fail，
//     实际动作编队异步）。
//   - open() 特例：N12Z1 在起逻辑线程**前**发（ACK 经 rx 线程入环，逻辑线程
//     pump 消化）；close() 特例：停线程后余任务丢弃（退出场景不保送）。
//   - 参数双口：getParam 读互斥保护快照（logicTick 每拍全量拷≤6 项）；
//     setParam 编队调账本 setValue。params() 引用出口已删（跨线程直调=竞态）。
//
// 逻辑线程一拍（logicTick，!manualTick 时 10ms 循环）：
//   drain 任务队列 → pump 上行环 → 按键手势 drain/dispatch → CommandChannel
//   对账 tick（经 MCUDriver::channelTick——S-T5 口径：pump 不调 tick，归本类驱动）
//   → Warmup tick → v2 温度断流兜底 → 采集中相机掉线巡检（只报不停）→ 参数快照。
// 切模式 = 命令组步链（sendSeq：前一条 ACK 完成回调里发下一条；任一步 3 败
// 整组短路+Fault）；组成功回调才擦板（ModeController「命令成功后才落板」）。
// Fault 出口统一 publishFault：EventBus FaultOccurred + spdlog warn。
//
// 口径裁定（D-T12b，未见计划明文的细节）：
//   - ctor 第 5 参 serialWriteOverride：MCUDriver 测试缝透传（std::function 直传，
//     不泄 MCUDriver 类型；产线空=真串口）；
//   - testInjectRaw：门面级测试缝（等价 rx 线程收到原始字节——MockMcu 回灌 ACK）；
//   - menuSelect ③/④ 出口：EventBus UserDefined（param1=3/4）——**待 base 增
//     专用事件类型（人工过会后改）**；app 订阅派工作流，门面只广播；①/② 仅
//     日志（modeCursor 账本为准）；
//   - 模式落板广播：EventBus StateChanged（param1=新态，替代直接 onChange 出口）；
//   - ParamStore onParamChanged → EventBus UserDefined（param1=参数索引=specs
//     登记序号，sourceId=8 标 08 来源）；
//   - 按键门禁（KeySemantics gate）= !isCapturing（M1：采集态菜单/模式/调节键
//     丢弃；启停键不问门禁）；
//   - N10 组参 Dispatch：采集中全参重发；空闲仅记账 done(true,false)（enterScan
//     时自账本组帧下发）；exposure 相机直设，无相机=纯记账 done(true,true)；
//   - v2 温度兜底周期 1200ms（发后重臂），首个 T 帧到达后才武装。
// ============================================================================

#include "IScannerCamera.h"
#include "KeyManager.h"       // GestureThresholds（DeviceConfig 值成员）
#include "MenuLogic.h"        // MenuState（menuState 返回值）
#include "ModeController.h"   // DeviceMode（mode 返回值）
#include "ParamStore.h"       // ParamEntry（getParam 返回值）
#include "WarmupSequence.h"   // WarmupConfig（DeviceConfig 值成员）
#include "base/EventBus.h"
#include "base/types.h"
#include "serial/FrameCodec.h"
#include "serial/McuFrame.h"  // TempFrame（getLastTemperatures 返回值）

#include <atomic>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace Scanner::device {

class MCUDriver;      // 子零件仅前向声明（铁规：不漏零件类型）
class KeySemantics;

struct DeviceConfig {
    std::string serialPort;
    int baud = 115200;
    serial::FrameCodec::Version protocol = serial::FrameCodec::Version::V3;
    int ackTimeoutMs = 100;
    GestureThresholds keys{};
    WarmupConfig warmup{};
    bool manualTick = false;      // 测试：不起逻辑线程，logicTick() 手动驱动
};

class DeviceManager {
public:
    using GateQuery = std::function<Result(const std::string& op)>;
    using CameraFactory = std::function<std::unique_ptr<hal::IScannerCamera>()>;
    // 串口写测试缝（MCUDriver 透传；std::function 直传不泄子零件类型）
    using SerialWriteOverride = std::function<bool(const std::string& frameBytes)>;

    DeviceManager(DeviceConfig cfg, GateQuery gate, infra::EventBus* bus,
                  CameraFactory camFactory = nullptr,
                  SerialWriteOverride serialWriteOverride = nullptr);
    ~DeviceManager();
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    // —— 开闭（open 一条龙：相机→MCU→上行接线→参数装载+快照→N12Z1→逻辑线程→
    //    DeviceConnected；任一同步步败：倒序关已开+Fault+fail）——
    Result open();
    Result close();

    // —— 切模式三口令（门禁调用方线程同步判；过→命令组编队逻辑线程执行，
    //    组成功回调才擦板；返回 request 结果）——
    Result enterCalibration();
    Result enterScan();
    Result toIdle();

    // —— 预热（N14 T<目标> + WarmupSequence 看火；稳/超时 done 恰一次、逻辑
    //    线程触发；超时不停加热——停止机制待协议方 §8-13）——
    void startWarmup(int targetC, std::function<void(bool stable)> done);

    // —— 观测 ——
    bool isDeviceReady() const;                 // 相机开+MCU 开（无相机工厂=只看 MCU）
    serial::TempFrame getLastTemperatures() const;
    bool isCapturing() const;                   // 采集子态真相源（ModeController 黑板）
    DeviceMode mode() const;                    // 模式黑板快照（UI/测试）
    MenuState menuState() const;                // 菜单账本快照（UI 常显）

    // —— 相机薄转发（统一编队：返回值=前置检查，实际动作逻辑线程异步执行）——
    bool isCameraOpen() const;
    Result setCameraExposure(double ms);
    Result startFrameStream(hal::FrameCallback cb);
    Result stopFrameStream();

    // —— 采集启停（N11 H1/H0 + 相机开停流；不切模式；幂等：黑板同值直返；
    //    编队执行——调用后需一拍 logicTick 落地下行帧）——
    void startCapture();
    void stopCapture();

    // —— 参数双口（Critical #1：账本归逻辑线程，引用出口已删）——
    ParamEntry getParam(const std::string& key) const;   // 互斥快照读（≤6 项轻拷）
    void setParam(const std::string& key, double v, ParamEntry::Source src);  // 编队 setValue

    void logicTick();                           // 逻辑线程主体一拍（manualTick 下测试驱动）
    void testInjectRaw(const std::string& frameBytes);   // 测试缝：等价 rx 收到原始字节

private:
    // 与 hal::IMCU::DoneCb 结构一致的下行命令完成回调（不经 IMCU.h——不泄子零件头）
    using McuDone = std::function<void(bool ok, const std::string& payload)>;
    struct SeqStep {                            // 命令组一步：描述 + 发送器
        std::string desc;
        std::function<void(McuDone)> send;
    };

    void logicLoop();                           // 10ms 循环调 logicTick
    void post(std::function<void()> task);      // 跨线程编队（容量 64 满丢新+warn）
    void drainPosts();                          // logicTick 开头排空（逻辑线程属主）
    void publishFault(int64_t code, const std::string& detail);
    void publishEvent(EventType t, int64_t p1, int64_t p2);
    void dispatchKeyGesture(const KeyGesture&);
    void buildKeyActions();                     // KeySemActions 11 出口的接线
    void applyAdjust(int dir);                  // 调节步进 → MenuLogic+ParamStore
    void sendSeq(std::vector<SeqStep> steps, std::function<void(bool)> onDone);
    void onParamDispatch(const std::string& key, double v, ParamStore::Done done);
    void startStreamIfReady();
    void refreshParamSnapshot();                // 全参数拷入互斥快照（≤6 项）
    // 切模式/启停的命令组主体（逻辑线程执行——门禁已在调用方线程过）
    void enterScanOnLogic();
    void enterCalibrationOnLogic();
    void toIdleOnLogic();
    void startCaptureOnLogic();
    void stopCaptureOnLogic();

    // —— 配置与依赖（声明序即初始化序）——
    DeviceConfig cfg_;
    infra::EventBus* bus_;
    CameraFactory camFactory_;
    SerialWriteOverride writeOverride_;

    // —— 组合零件（全内部持有不外泄；MCUDriver/KeySemantics 前向声明）——
    std::unique_ptr<hal::IScannerCamera> camera_;
    std::unique_ptr<MCUDriver> mcu_;
    std::unique_ptr<KeyManager> keyMgr_;
    std::unique_ptr<MenuLogic> menu_;
    std::unique_ptr<ModeController> mode_;
    std::unique_ptr<WarmupSequence> warmup_;
    std::unique_ptr<KeySemantics> semantics_;   // ctor 体内 buildKeyActions 接线
    std::unique_ptr<ParamStore> params_;        // ctor 体内接线（dispatch 捕 this）

    // —— 编队队列（入队任意线程 / 消费逻辑线程）——
    std::mutex postMutex_;
    std::deque<std::function<void()>> postQueue_;

    // —— 参数快照（逻辑线程写 / 任意线程读，互斥保护）——
    std::vector<std::string> paramKeys_;        // specs 登记序（param1 索引线索）
    mutable std::mutex snapshotMutex_;
    std::map<std::string, ParamEntry> paramSnapshot_;

    // —— 运行时记账（逻辑线程属主；const 读口走原子/值拷贝）——
    hal::FrameCallback frameCb_;                // startFrameStream 登记的帧出口
    serial::TempFrame lastTemps_{};
    TimestampMs tempRxTime_ = 0;                // 最近 T 帧到达时刻（v2 兜底判据）
    std::function<void(bool)> warmupDone_;      // 当前预热完成回调（onStable/onTimeout 消费）
    bool standbyActive_ = false;                // MCU 侧待机记账（切模式前退待机判据）
    bool camFaultLatched_ = false;              // 掉线故障边沿锁（复报抑制）
    bool opened_ = false;
    std::atomic<bool> running_{false};
    std::thread logicThread_;
};

} // namespace Scanner::device
