#pragma once
// ============================================================================
// DeviceManager.h — 设备管理门面 + 逻辑线程（D-T12b；设计方案 §4 五增量）
//
// 组合根：MCUDriver/KeyManager/KeySemantics/MenuLogic/ParamStore/WarmupSequence/
// ModeController 全 unique_ptr 内部持有（铁规：不漏零件指针——对外只出值类型
// 与薄转发）。门禁回调注入不反链 07/10（GateQuery 只进不出）。
//
// 逻辑线程（!manualTick 时 10ms 循环；manualTick=true 测试手动驱动 logicTick）：
//   pump 上行环 → 按键手势 drain/dispatch → CommandChannel 对账 tick（经
//   MCUDriver::channelTick——S-T5 口径：pump 不调 tick，归本类驱动）→ Warmup
//   tick → v2 温度断流兜底 → 采集中相机掉线巡检（只报不停）。
// 切模式 = 命令组步链（sendSeq：前一条 ACK 完成回调里发下一条；任一步 3 败
// 整组短路+Fault）；组成功回调才擦板（ModeController「命令成功后才落板」）。
// Fault 出口统一 publishFault：EventBus FaultOccurred + spdlog warn。
//
// 口径裁定（D-T12b，未见计划明文的细节）：
//   - ctor 第 5 参 serialWriteOverride：MCUDriver 测试缝透传（产线空=真串口）；
//   - testInjectRaw：门面级测试缝（等价 rx 线程收到原始字节——MockMcu 回灌 ACK）；
//   - menuSelect ③/④ 出口：EventBus ScanStopped / PostProcessStarted（派工作流
//     归 app 订阅，门面只广播）；①/② 仅日志（modeCursor 账本为准）；
//   - 模式落板广播：EventBus StateChanged（param1=新态，替代直接 onChange 出口）；
//   - ParamStore onParamChanged → EventBus UserDefined（base 暂无 ParamChanged
//     事件类型，sourceId=8 标来源）；
//   - 按键门禁（KeySemantics gate）= !isCapturing（M1：采集态菜单/模式/调节键
//     丢弃；启停键不问门禁）；
//   - N10 组参 Dispatch：采集中全参重发；空闲仅记账 done(true,false)（enterScan
//     时自账本组帧下发）；exposure 相机直设，无相机=纯记账 done(true,true)；
//   - v2 温度兜底周期 1200ms（发后重臂），首个 T 帧到达后才武装。
// ============================================================================

#include "IScannerCamera.h"
#include "KeyManager.h"
#include "KeySemantics.h"
#include "MCUDriver.h"
#include "MenuLogic.h"
#include "ModeController.h"
#include "ParamStore.h"
#include "WarmupSequence.h"
#include "base/EventBus.h"
#include "base/types.h"
#include "serial/FrameCodec.h"
#include "serial/McuFrame.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace Scanner::device {

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

    // serialWriteOverride：MCUDriver 测试缝透传（非空=不开真串口不起 rx 线程）
    DeviceManager(DeviceConfig cfg, GateQuery gate, infra::EventBus* bus,
                  CameraFactory camFactory = nullptr,
                  MCUDriver::WriteOverride serialWriteOverride = nullptr);
    ~DeviceManager();
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    // —— 开闭（open 一条龙：相机→MCU→上行接线→参数装载→逻辑线程→N12Z1→
    //    DeviceConnected；任一同步步败：倒序关已开+Fault+fail）——
    Result open();
    Result close();

    // —— 切模式三口令（命令组步链；组成功回调才擦板）——
    void enterCalibration();
    void enterScan();
    void toIdle();

    // —— 预热（N14 T<目标> + WarmupSequence 看火；稳/超时 done 恰一次；
    //    超时不停加热——停止机制待协议方 §8-13）——
    void startWarmup(int targetC, std::function<void(bool stable)> done);

    // —— 观测 ——
    bool isDeviceReady() const;                 // 相机开+MCU 开（无相机工厂=只看 MCU）
    serial::TempFrame getLastTemperatures() const;
    bool isCapturing() const;                   // 采集子态真相源（ModeController 黑板）
    DeviceMode mode() const;                    // 模式黑板快照（UI/测试）
    MenuState menuState() const;                // 菜单账本快照（UI 常显）

    // —— 相机薄转发（窗口不漏零件，值/Result 转发允许）——
    bool isCameraOpen() const;
    Result setCameraExposure(double ms);
    Result startFrameStream(hal::FrameCallback cb);
    Result stopFrameStream();

    // —— 采集启停（N11 H1/H0 + 相机开停流；不切模式；幂等：黑板同值直返）——
    void startCapture();
    void stopCapture();

    ParamStore& params();                       // UI 双向同步入口（唯一真相源）

    void logicTick();                           // 逻辑线程主体一拍（manualTick 下测试驱动）
    void testInjectRaw(const std::string& frameBytes);   // 测试缝：等价 rx 收到原始字节

private:
    using McuDone = hal::IMCU::DoneCb;
    struct SeqStep {                            // 命令组一步：描述 + 发送器
        std::string desc;
        std::function<void(McuDone)> send;
    };

    void logicLoop();                           // 10ms 循环调 logicTick
    void publishFault(int64_t code, const std::string& detail);
    void publishEvent(EventType t, int64_t p1, int64_t p2);
    void dispatchKeyGesture(const KeyGesture&);
    void buildKeyActions();                     // KeySemActions 11 出口的接线
    void applyAdjust(int dir);                  // 调节步进 → MenuLogic+ParamStore
    void sendSeq(std::vector<SeqStep> steps, std::function<void(bool)> onDone);
    hal::CaptureParams captureParamsFromAccount() const;
    void onParamDispatch(const std::string& key, double v, ParamStore::Done done);
    void startStreamIfReady();

    // —— 配置与依赖（声明序即初始化序）——
    DeviceConfig cfg_;
    infra::EventBus* bus_;
    CameraFactory camFactory_;
    MCUDriver::WriteOverride writeOverride_;

    // —— 组合零件（全内部持有不外泄）——
    std::unique_ptr<hal::IScannerCamera> camera_;
    std::unique_ptr<MCUDriver> mcu_;
    std::unique_ptr<KeyManager> keyMgr_;
    std::unique_ptr<MenuLogic> menu_;
    std::unique_ptr<ModeController> mode_;
    std::unique_ptr<WarmupSequence> warmup_;
    std::unique_ptr<KeySemantics> semantics_;   // ctor 体内 buildKeyActions 接线
    std::unique_ptr<ParamStore> params_;        // ctor 体内接线（dispatch 捕 this）

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
