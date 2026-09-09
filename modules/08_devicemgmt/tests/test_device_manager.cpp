// ============================================================================
// test_device_manager.cpp — DeviceManager 门面集成测（协议 260831 批1-C 适配）
//
// 全链真件（MCUDriver/KeySemantics/MenuLogic/ParamStore/Warmup/ModeController
// 全真配），假件仅两处边界：
//   - MockMcu：writeOverride 记下行帧（盲发无 ACK——无回灌机制）；
//   - FakeCamera：IScannerCamera 全接口空壳，isOpen 可拨（掉线模拟）。
// manualTick=true：不起逻辑线程，logicTick() 手动驱动。上行帧经 testInjectRaw
// 回灌（"G01 U1;"=上键短按——手势判定归 MCU G01，PC 侧无判定；KeyManager 已退役）。
// 用例编号承接 D-T12b T1–T14 + D-T13 F1–F7：T10（ACK 重传）/T11（v2 兜底）/
// F7（seq 跳变）随 ACK/seq 机制退役删（批2 补直发版）；T9 改 close/reopen 无残留；
// T12 改 enterScan 单步 N10 落板；T14 改 enterCalibration 纯软件落板。
// 批4 补：T9b/T9c（D8 恢复路径——同实例 reopen 清账＋重开重同步分支）；F2 改
// open 即武装（serialArmed_）；F4 改绝对差值判据（tempSpikeAbsC）。
// ============================================================================

#include <gtest/gtest.h>

#include "modules/08_devicemgmt/DeviceManager.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace Scanner::device;
using Scanner::Event;
using Scanner::EventType;
using Scanner::Result;

namespace {

// 账本默认值组帧的 N10（freqHz 60 / bgLight 10 / laserLevel 40 / T1V1C0D0）
const char* kN10Default = "N10 H60 B10 T1 V1 C0 D0 L40";

void sleepMs(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

auto gateOk = [](const std::string&) { return Result::ok(); };
auto gateReject = [](const std::string& op) { return Result::fail(1, "门禁拒绝:" + op); };

// —— 事件记账（subscribeAll 同步回调；互斥保护——T13 双线程并发 publish）——
struct EventRecorder {
    mutable std::mutex m;
    std::vector<Event> ev;
    void record(const Event& e) {
        std::lock_guard<std::mutex> lock(m);
        ev.push_back(e);
    }
    int count(EventType t) const {
        std::lock_guard<std::mutex> lock(m);
        int n = 0;
        for (const auto& e : ev)
            if (e.type == t) ++n;
        return n;
    }
    // UserDefined 按 param1 计数（menuSelect ③/④ 出口）
    int userParam(int64_t p1) const {
        std::lock_guard<std::mutex> lock(m);
        int n = 0;
        for (const auto& e : ev)
            if (e.type == EventType::UserDefined && e.param1 == p1) ++n;
        return n;
    }
    // Fault 按码计数（D-T13：DevFault 码表断言）
    int fault(int64_t faultCode) const {
        std::lock_guard<std::mutex> lock(m);
        int n = 0;
        for (const auto& e : ev)
            if (e.type == EventType::FaultOccurred && e.param1 == faultCode) ++n;
        return n;
    }
};

// DevFault 码 → int64（断言简写）
constexpr int64_t FC(DevFault f) { return static_cast<int64_t>(f); }

// —— 假相机：全接口空壳 + isOpen 可控（T8/F1 掉线模拟）——
struct FakeCamera : Scanner::hal::IScannerCamera {
    bool openOk = true;
    bool openState = false;
    double exposureMs = 0.0;
    int exposureCalls = 0;

    std::string getDeviceName() const override { return "FakeCamera"; }
    std::string getSerialNumber() const override { return "FAKE-001"; }
    Result open() override {
        if (!openOk) return Result::fail("相机打开失败(测试)");
        openState = true;
        return Result::ok();
    }
    Result close() override { openState = false; return Result::ok(); }
    bool isOpen() const override { return openState; }
    Result setExposure(double ms) override {
        ++exposureCalls;
        exposureMs = ms;
        return Result::ok();
    }
    Result setGain(double) override { return Result::ok(); }
    Result setResolution(int, int) override { return Result::ok(); }
    Result setCalibration(const Scanner::hal::CameraIntrinsics&,
                          const Scanner::hal::CameraIntrinsics&,
                          const Scanner::hal::StereoExtrinsics&) override { return Result::ok(); }
    Scanner::hal::CameraIntrinsics getLeftIntrinsics() const override { return {}; }
    Scanner::hal::CameraIntrinsics getRightIntrinsics() const override { return {}; }
    Scanner::hal::StereoExtrinsics getStereoExtrinsics() const override { return {}; }
    Result startCapture() override { return Result::ok(); }
    Result stopCapture() override { return Result::ok(); }
    bool isCapturing() const override { return false; }
    Result grabFrame(Scanner::hal::StereoFrame&, int) override { return Result::fail("未实现"); }
    Result startAsyncCapture(Scanner::hal::FrameCallback) override {
        return openState ? Result::ok() : Result::fail("相机未开");
    }
    Result stopAsyncCapture() override { return Result::ok(); }
    double getTemperature() const override { return 0.0; }
    std::string getPlatform() const override { return "Windows"; }
};

// —— 假 MCU：记全部下行帧（盲发无 ACK——写即成功，无回灌）——
struct MockMcu {
    DeviceManager* dm = nullptr;                  // 接缝保留（回灌口径批2 复核）
    std::vector<std::string> frames;

    bool write(const std::string& f) {
        frames.push_back(f);
        return true;
    }
    int count(const std::string& sub) const {
        int n = 0;
        for (const auto& f : frames)
            if (f.find(sub) != std::string::npos) ++n;
        return n;
    }
};

// —— 测试配置（manualTick + 小阈值换快用例）——
DeviceConfig makeCfg() {
    DeviceConfig c;
    c.serialPort = "COM_TEST";
    c.baud = 115200;
    c.tempReportPeriodMs = 100;
    c.warmup = WarmupConfig{100, 0.1, 2.0, 3000};             // 稳定窗 100ms
    c.manualTick = true;
    return c;
}

// —— 上行注入工具（裸 ';' 帧；手势=G01 单帧——MCU 已判 S/D/H）——
struct Kit {
    DeviceManager* dm = nullptr;

    void raw(const std::string& payload) { dm->testInjectRaw(payload + ";"); }
    void gesture(char key, int g) {              // g：1=短按 2=双击 3=长按
        raw("G01 " + std::string{key} + std::to_string(g));
        dm->logicTick();                         // pump+派发同拍落地
    }
    void gShort(char k) { gesture(k, 1); }
    void gDouble(char k) { gesture(k, 2); }
    void gHold(char k) { gesture(k, 3); }
    void temp(double c) {                        // G02 四路同值（双警/Warmup 取 celsius[0]）
        const std::string v = std::to_string(c);
        raw("G02 A" + v + " B" + v + " C" + v + " D" + v);
        dm->logicTick();
    }
};

} // namespace

// —— T1：相机打开失败 → open fail + 倒序关闭无崩 + Fault 事件（MCU 未开无命令帧）——
TEST(DeviceManager, T1_OpenFailCameraRollbackAndFault) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus,
                     [] {
                         auto c = std::make_unique<FakeCamera>();
                         c->openOk = false;
                         return c;
                     },
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;

    const Result r = dm.open();
    EXPECT_FALSE(r.success);
    EXPECT_GE(rec.count(EventType::FaultOccurred), 1);
    EXPECT_FALSE(dm.isCameraOpen());
    EXPECT_TRUE(mock.frames.empty());                          // 相机败先返：N12 都没发
}                                                              // 析构倒序收尾——无崩即过

// —— T2：门禁拒切扫描 → enterScan 返回后无任何命令组下行帧 ——
TEST(DeviceManager, T2_GateRejectEnterScanNoFrames) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateReject, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    EXPECT_FALSE(dm.enterScan().success);                     // 门禁拒：同步返回 fail（不入队）
    EXPECT_EQ(mock.count("N10"), 0);
    EXPECT_EQ(mock.count("N11"), 0);
    EXPECT_EQ(mock.count("N13"), 0);
    dm.logicTick();                                            // 补一拍证明确无任务落地
    ASSERT_EQ(mock.frames.size(), 1u);                        // 仅 open 的 N12 T100
    EXPECT_EQ(mock.frames[0], "N12 T100;");
    EXPECT_EQ(rec.count(EventType::StateChanged), 0);          // 黑板未动
}

// —— T3：预热升温序列 → 稳定回调恰一次（双点锚点法）——
TEST(DeviceManager, T3_WarmupStableCallbackOnce) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    Kit kit;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    kit.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    int cbCount = 0;
    bool cbVal = false;
    dm.startWarmup(42, [&](bool stable) {
        ++cbCount;
        cbVal = stable;
    });
    dm.logicTick();                                            // 编队任务落地（N13 S42 下发）
    EXPECT_EQ(mock.count("N13 S42"), 1);                       // 加热命令已发

    kit.temp(20.0);
    sleepMs(30);
    kit.temp(35.0);
    sleepMs(30);
    kit.temp(41.5);
    sleepMs(30);
    kit.temp(42.0);                                            // 平台锚点
    sleepMs(120);
    kit.temp(42.0);                                            // 窗满(120≥100)+不动(0≤0.1)+近目标(0≤2) → 稳
    EXPECT_EQ(cbCount, 1);
    EXPECT_TRUE(cbVal);
    kit.temp(42.0);                                            // Done 后不再回调
    EXPECT_EQ(cbCount, 1);
}

// —— T4：预热超时回调恰一次 + 无「停止加热」下行帧（只报不停——协议未定）——
TEST(DeviceManager, T4_WarmupTimeoutCallbackOnceNoStopHeat) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    cfg.warmup.timeoutMs = 200;
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    int cbCount = 0;
    bool cbVal = true;
    dm.startWarmup(42, [&](bool stable) {
        ++cbCount;
        cbVal = stable;
    });
    for (int i = 0; i < 60 && cbCount == 0; ++i) {
        sleepMs(25);
        dm.logicTick();
    }
    EXPECT_EQ(cbCount, 1);
    EXPECT_FALSE(cbVal);
    EXPECT_EQ(mock.count("N13 S42"), 1);
    EXPECT_EQ(mock.count("N13 S0"), 0);                         // 超时不停加热
}

// —— T5：中键短按启停（启=N10 停=N11 H0 按黑板）+ 直调幂等（连按同值不乱）——
TEST(DeviceManager, T5_CaptureToggleByIdempotent) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    Kit kit;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    kit.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    kit.gShort('M');                                           // 主层中键短按 → 启采集（单步 N10=启采）
    EXPECT_EQ(mock.count(kN10Default), 1);
    EXPECT_EQ(mock.count("N11 H1"), 0);                        // 260831：无 N11 H1
    EXPECT_TRUE(dm.isCapturing());
    kit.gShort('M');                                           // 再按 → 停采集（单发 N11 H0）
    EXPECT_EQ(mock.count("N11 H0"), 1);
    EXPECT_FALSE(dm.isCapturing());

    dm.startCapture();                                         // 直调重复启
    dm.logicTick();
    EXPECT_EQ(mock.count(kN10Default), 2);                     // 盲发即回调——一拍内完成
    EXPECT_TRUE(dm.isCapturing());
    dm.startCapture();                                         // 采集已开：幂等无新 N10
    dm.logicTick();
    EXPECT_EQ(mock.count(kN10Default), 2);
    dm.stopCapture();                                          // 直调重复停：幂等无新帧
    dm.logicTick();
    dm.stopCapture();
    dm.logicTick();
    EXPECT_EQ(mock.count("N11 H0"), 2);
    EXPECT_FALSE(dm.isCapturing());
}

// —— T5b（A-T17 钉死·260831 适配）：setParam 改账后 startCapture →
//      N10 全参自 ParamStore 账本组帧（非 MCU 默认参数）——
TEST(DeviceManager, T5b_StartCaptureN10FromParamAccount) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    dm.setParam("freqHz", 90.0, ParamEntry::Source::Ui);       // 空闲改账（纯记账）
    dm.logicTick();
    dm.startCapture();
    dm.logicTick();
    EXPECT_EQ(mock.count("N10 H90 B10 T1 V1 C0 D0 L40"), 1);   // N10 帧含 H90（账本值）
    EXPECT_EQ(mock.count("N11 H1"), 0);                        // 无 N11 H1（启采=N10 本身）
    EXPECT_TRUE(dm.isCapturing());
}

// —— T5c（协议批3·D7）：四模式 N10 四管掩码映射——MarkerOnly 不开激光线
//      （B=40 抬升基线/L=0/四管全 0）；面片 T1V1；精细 C 管；深孔 D 管
//      （账本默认 freq 60/bg 10/laser 40；⑨b：单管周期配对归 07 侧待裁决）——
TEST(DeviceManager, T5c_FourModeN10TubeMasks) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    dm.startCapture(Scanner::ScanMode::FineScan);                // 精细：C 管
    dm.logicTick();
    EXPECT_EQ(mock.count("N10 H60 B10 T0 V0 C1 D0 L40"), 1);
    EXPECT_TRUE(dm.isCapturing());
    dm.stopCapture();
    dm.logicTick();

    dm.startCapture(Scanner::ScanMode::DeepHoleScan);            // 深孔：D 管
    dm.logicTick();
    EXPECT_EQ(mock.count("N10 H60 B10 T0 V0 C0 D1 L40"), 1);
    dm.stopCapture();
    dm.logicTick();

    dm.startCapture(Scanner::ScanMode::MarkerOnly);              // 标点：B=40 抬升+全管灭
    dm.logicTick();
    EXPECT_EQ(mock.count("N10 H60 B40 T0 V0 C0 D0 L0"), 1);
    dm.stopCapture();
    dm.logicTick();
    EXPECT_EQ(mock.count("N11 H0"), 3);                          // 三轮启停各一帧收口
}

// —— T6：菜单全遍历（4 键×3 手势）—— layer2/游标环绕/调节上下文/模式光标可达性 ——
TEST(DeviceManager, T6_MenuTraversalFourKeysThreeGestures) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    Kit kit;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus,
                     [] { return std::make_unique<FakeCamera>(); },
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    kit.dm = &dm;
    ASSERT_TRUE(dm.open().success);
    auto st = [&] { return dm.menuState(); };

    EXPECT_EQ(st().layer, 1);
    kit.gShort('U');                                           // 上键短按 L1：进菜单（cursor 复位①）
    EXPECT_EQ(st().layer, 2);
    EXPECT_EQ(st().cursor, 1);
    for (int i = 0; i < 4; ++i) kit.gShort('R');               // 右键短按×4：1→2→3→4→1 环绕
    EXPECT_EQ(st().cursor, 1);
    kit.gShort('L');                                           // 左键短按：1→4 环绕
    EXPECT_EQ(st().cursor, 4);
    kit.gDouble('M');                                          // 中键双击：模式光标 3→1→2→3
    EXPECT_EQ(st().modeCursor, 1);
    kit.gDouble('M');
    EXPECT_EQ(st().modeCursor, 2);
    kit.gDouble('M');
    EXPECT_EQ(st().modeCursor, 3);
    const int post0 = rec.userParam(4);
    kit.gShort('M');                                           // 中键短按 L2 选中④：派后处理工作流（UserDefined p1=4）
    EXPECT_EQ(rec.userParam(4), post0 + 1);
    kit.gShort('U');                                           // 上键短按 L2：退菜单
    EXPECT_EQ(st().layer, 1);

    kit.gDouble('U');                                          // 上键双击：None→View
    EXPECT_EQ(st().adjustCtx, MenuState::AdjustCtx::View);
    kit.gShort('R');                                           // View 上下文：暂仅日志（曝光不动）
    const double base = dm.getParam("exposure").value;
    kit.gDouble('U');                                          // View→Brightness
    EXPECT_EQ(st().adjustCtx, MenuState::AdjustCtx::Brightness);
    kit.gShort('R');                                           // 右键短按：曝光 +1ms（相机直设）
    EXPECT_DOUBLE_EQ(dm.getParam("exposure").value, base + 1.0);
    kit.gShort('L');                                           // 左键短按：曝光 -1ms
    EXPECT_DOUBLE_EQ(dm.getParam("exposure").value, base);
    kit.gDouble('U');                                          // Brightness→None
    EXPECT_EQ(st().adjustCtx, MenuState::AdjustCtx::None);

    kit.gShort('L');                                           // 主层无上下文左右：无效丢弃
    EXPECT_EQ(st().layer, 1);
    kit.gDouble('L');                                          // 双击/长按协议可达未定义手势全丢弃不崩
    kit.gDouble('R');
    kit.gHold('U');
    kit.gHold('M');
    kit.gHold('L');
    kit.gHold('R');
    EXPECT_EQ(st().layer, 1);
    // 快照一致性：末拍刷新后 menuState()（互斥快照口）= 逻辑线程账本状态
    EXPECT_EQ(dm.menuState().layer, 1);
    EXPECT_EQ(dm.menuState().adjustCtx, MenuState::AdjustCtx::None);
}

// —— T7：手势洪峰 100 帧 → 手势环有效容量 63（SpscRing<64> 满丢新）收敛——
//      63 个 U1 手势全派发（进/出菜单交替——幂等不崩）、无崩溃 ——
TEST(DeviceManager, T7_KeyFlood100NoCrash) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    Kit kit;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    kit.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    for (int i = 0; i < 100; ++i) kit.raw("G01 U1;");          // 100 手势帧（上键短按）
    dm.logicTick();                                            // 环容 64：仅前 63 入环（37 丢新）
    // 63 个 U1 手势同拍全派发：1st 进菜单→2nd 退菜单…（奇数 63 → 终态 layer 2）；
    // U 键无采集副作用；溢出 Fault 钉死在 F6——本测只证洪峰不崩
    EXPECT_EQ(mock.count("N10 ") + mock.count("N11 H0"), 0);
    EXPECT_EQ(dm.menuState().layer, 2);
    dm.logicTick();                                            // 再拍无残留不崩
    EXPECT_EQ(dm.menuState().layer, 2);
}

// —— T8：采集中相机掉线 → Fault 且无自主停采（只报不动手：无 N11 H0）——
TEST(DeviceManager, T8_CameraDisconnectDuringCaptureFaultNoAutoStop) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    FakeCamera* fake = nullptr;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus,
                     [&] {
                         auto c = std::make_unique<FakeCamera>();
                         fake = c.get();
                         return c;
                     },
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    dm.startCapture();
    dm.logicTick();
    ASSERT_TRUE(dm.isCapturing());

    fake->openState = false;                                   // 相机掉线
    dm.logicTick();                                            // 下一拍巡检点
    EXPECT_GE(rec.count(EventType::FaultOccurred), 1);
    EXPECT_EQ(mock.count(kN10Default), 1);                     // 启采那帧 N10
    EXPECT_EQ(mock.count("N11 H0"), 0);                        // 无自主停采
    EXPECT_FALSE(dm.isDeviceReady());
}

// —— T9：close→reopen 无残留（原 v2/v3 切换用例随协议版本退役删）——
//      手势链活 → close → reopen：N12 重发、无残留采集态、手势链仍活 ——
TEST(DeviceManager, T9_CloseReopenGestureChainAlive) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    Kit kit;
    auto write = [&](const std::string& f) { return mock.write(f); };

    {
        DeviceConfig cfg = makeCfg();
        DeviceManager dm(cfg, gateOk, &bus, nullptr, write);
        mock.dm = &dm;
        kit.dm = &dm;
        ASSERT_TRUE(dm.open().success);
        kit.gShort('M');                                       // 启采集（手势链活）
        EXPECT_TRUE(dm.isCapturing());
        ASSERT_TRUE(dm.close().success);                       // close：N11 H0 收口+状态复位
        EXPECT_FALSE(dm.isCapturing());                        // D8 最小版：黑板复位
        EXPECT_EQ(mock.count("N11 H0"), 1);
    }                                                          // 析构 close 幂等

    DeviceConfig cfg2 = makeCfg();
    DeviceManager dm2(cfg2, gateOk, &bus, nullptr, write);
    mock.dm = &dm2;
    kit.dm = &dm2;
    ASSERT_TRUE(dm2.open().success);
    EXPECT_EQ(mock.count("N12 T100"), 2);                      // 两次 open 各下发一次
    EXPECT_FALSE(dm2.isCapturing());                           // 无残留采集态
    kit.gShort('M');                                           // 手势链仍活
    EXPECT_TRUE(dm2.isCapturing());
}

// —— T9b（批4·D8 恢复路径）：同实例 close→reopen——N12 重发、旧温清零（ts==0，
//      含快照同步）、黑板 Idle/非采集、无自动 N10 重同步（重开重同步仅在黑板非
//      Idle 时触发——见 T9c）；采集中 close → reopen 同样收敛（不复活采集）——
TEST(DeviceManager, T9b_CloseReopenRecoveryPath) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    Kit kit;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                      [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    kit.dm = &dm;
    ASSERT_TRUE(dm.open().success);
    kit.temp(30.0);                                            // 旧会话温度（lastTemps_ 落账 ts>0）
    EXPECT_GT(dm.getLastTemperatures().ts, 0);                 // 前提：旧温可见

    ASSERT_TRUE(dm.close().success);                           // close：N11 H0+黑板复位+旧温清零
    EXPECT_EQ(dm.getLastTemperatures().ts, 0);                 // 旧温清零（快照同拍归零）
    EXPECT_FALSE(dm.isCapturing());
    EXPECT_EQ(dm.mode(), DeviceMode::Idle);

    ASSERT_TRUE(dm.open().success);                            // reopen（同实例）
    EXPECT_EQ(mock.count("N12 T100"), 2);                      // N12 重发（两会话各一）
    EXPECT_FALSE(dm.isCapturing());                            // 无残留采集态
    EXPECT_EQ(dm.mode(), DeviceMode::Idle);
    EXPECT_EQ(mock.count("N10 "), 0);                          // 黑板 Idle：无 N10 重同步帧
    EXPECT_EQ(dm.getLastTemperatures().ts, 0);                 // reopen 首帧前不回旧温

    // —— 采集中 close → reopen：同样收敛（MCU 收口靠 close 的 N11 H0，reopen 不
    //    自动重启采集——重开重同步分支不触发）——
    dm.startCapture(Scanner::ScanMode::FineScan);
    dm.logicTick();
    ASSERT_TRUE(dm.isCapturing());
    const int n10Before = mock.count("N10 ");
    ASSERT_TRUE(dm.close().success);
    ASSERT_TRUE(dm.open().success);
    EXPECT_FALSE(dm.isCapturing());
    EXPECT_EQ(dm.mode(), DeviceMode::Idle);
    EXPECT_EQ(mock.count("N12 T100"), 3);                      // 第三次 open 再重发
    EXPECT_EQ(mock.count("N10 "), n10Before);                  // 无新增 N10（Idle 黑板不重同步）
}

// —— T9c（批4·D8 恢复路径·重开重同步分支）：close 后黑板被外部置非 Idle（如重开
//      期间 UI 先行切扫描态）→ open 成功尾段按 lastCaptureMode_ 重发 N10 参数重同步
//      ——防御分支（正常 close→open 黑板已复位不触发，见 T9b）——
TEST(DeviceManager, T9c_ReopenResyncN10WhenBoardNonIdle) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                      [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);
    dm.startCapture(Scanner::ScanMode::FineScan);              // lastCaptureMode_=精细（跨 close 保留）
    dm.logicTick();
    ASSERT_TRUE(dm.isCapturing());
    ASSERT_TRUE(dm.close().success);                           // 黑板复位 Idle
    EXPECT_TRUE(dm.enterScan().success);                       // 设备关期间黑板再入扫描
    dm.logicTick();                                            // 落板（Scanning+capturing；N10 经 override 记帧）
    ASSERT_EQ(dm.mode(), DeviceMode::Scanning);
    const int n10Before = mock.count("N10 ");
    ASSERT_GE(n10Before, 2);                                   // startCapture+enterScan 各一

    ASSERT_TRUE(dm.open().success);                            // reopen：成功尾段见黑板非 Idle
    EXPECT_EQ(mock.count("N10 "), n10Before + 1);              // 重开重同步：补发一帧 N10
    EXPECT_EQ(mock.count("N10 H60 B10 T0 V0 C1 D0 L40"), 3);   // 三帧同精细灯型（账本默认参）
    EXPECT_EQ(mock.count("N12 T100"), 2);                      // N12 亦重发
    EXPECT_TRUE(dm.isCapturing());                             // 黑板语义保持（MCU 已同步）
}

// —— T12：enterScan 单步 N10 → 擦板+采集开+StateChanged 恰一次（原 ACK 组链
//      中段对照随无 ACK 机制退役删——组失败路径批2 直发版补）——
TEST(DeviceManager, T12_EnterScanSingleN10Commit) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    EXPECT_TRUE(dm.enterScan().success);                       // 门禁过（编队执行）
    dm.logicTick();
    EXPECT_EQ(mock.count(kN10Default), 1);                     // 启采=单步 N10（账本全参）
    EXPECT_EQ(mock.count("N11 H1"), 0);                        // 260831：无 N11 H1
    EXPECT_EQ(dm.mode(), DeviceMode::Scanning);
    EXPECT_TRUE(dm.isCapturing());
    EXPECT_EQ(rec.count(EventType::StateChanged), 1);          // commit(Scanning) 落板广播恰一次
}

// —— T13（Critical #1 回归）：双线程真并发冒烟——manualTick=false 起真逻辑线程，
//      另一线程连发 50 次 setParam+startCapture/stopCapture 交替（+并发 getParam
//      快照读），2s 后 close 停线程清队——无死锁无崩溃 ——
TEST(DeviceManager, T13_ConcurrentPostSmoke) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    cfg.manualTick = false;                                    // 真逻辑线程 10ms
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    std::atomic<bool> stopFlag{false};
    std::thread worker([&] {
        for (int i = 0; i < 50 && !stopFlag.load(); ++i) {
            dm.setParam("bgLight", 60.0 + (i % 40), ParamEntry::Source::Ui);
            dm.setParam("exposure", 20.0 + (i % 50), ParamEntry::Source::Ui);
            if (i % 2 == 0) dm.startCapture();
            else dm.stopCapture();
            (void)dm.getParam("exposure");                     // 并发快照读
            sleepMs(20);
        }
    });
    sleepMs(2000);                                             // 逻辑线程满速跑拍
    stopFlag.store(true);
    worker.join();
    EXPECT_TRUE(dm.close().success);                           // 停线程+清队+关 MCU 无死锁
    EXPECT_GE(mock.count("N10 ") + mock.count("N11 H0"), 1);   // 任务确有落地
    EXPECT_GE(dm.getParam("bgLight").value, 0.0);              // 快照口仍可读
}

// —— T14（Important #3 补缺·260831 适配）：enterCalibration 纯软件落板——
//      无任何新下行帧（原 N16 组链已删）+commit Calibrating+StateChanged 恰一次；
//      再入 same-mode commit 不广播（D-T9）——
TEST(DeviceManager, T14_EnterCalibrationPureSoftware) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);
    const size_t framesBefore = mock.frames.size();            // open 的 N12 之后

    EXPECT_TRUE(dm.enterCalibration().success);                // 门禁过（编队执行）
    dm.logicTick();
    EXPECT_EQ(mock.frames.size(), framesBefore);               // 纯软件落板：无任何新下行帧
    EXPECT_EQ(dm.mode(), DeviceMode::Calibrating);
    EXPECT_EQ(rec.count(EventType::StateChanged), 1);          // 落板广播恰一次

    EXPECT_TRUE(dm.enterCalibration().success);                // 再入：same-mode 不广播
    dm.logicTick();
    EXPECT_EQ(mock.frames.size(), framesBefore);
    EXPECT_EQ(rec.count(EventType::StateChanged), 1);
}

// ============================================================================
// D-T13：故障 8 码接线（设计方案 §6.2；边沿纪律=恢复清锚）
// ============================================================================

// —— F1（#1）：非采集中相机掉线 → 0x0801 恰一次；再拍不重复；恢复→再掉→再触发 ——
TEST(DeviceManager, F1_CameraLostAnyTimeEdge) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    FakeCamera* fake = nullptr;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus,
                     [&] {
                         auto c = std::make_unique<FakeCamera>();
                         fake = c.get();
                         return c;
                     },
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);
    ASSERT_FALSE(dm.isCapturing());                            // 全程非采集中

    fake->openState = false;                                   // 掉线（曾开→翻 false 边沿）
    dm.logicTick();
    EXPECT_EQ(rec.fault(FC(DevFault::CameraLost)), 1);
    dm.logicTick();                                            // 边沿锁：不重复
    EXPECT_EQ(rec.fault(FC(DevFault::CameraLost)), 1);

    fake->openState = true;                                    // 恢复 → 清锚
    dm.logicTick();
    fake->openState = false;                                   // 再掉 → 再触发
    dm.logicTick();
    EXPECT_EQ(rec.fault(FC(DevFault::CameraLost)), 2);
}

// —— F2（#2≡#10）：open 即武装（批4——旧口径武装门 lastRx>0 致「全程无帧」
//      永不告警）：无任何帧下超时 → 0x0802 边沿一次；恢复帧清锚后可再触发 ——
TEST(DeviceManager, F2_HeartbeatTimeoutArmedAtOpen) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    Kit kit;
    DeviceConfig cfg = makeCfg();
    cfg.heartbeatTimeoutMs = 100;                              // 测试注入：100ms 判无声
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                      [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    kit.dm = &dm;
    ASSERT_TRUE(dm.open().success);                            // open 成功即武装（无帧也巡检）

    dm.logicTick();                                            // 距武装 <100ms：窗口内无警
    EXPECT_EQ(rec.fault(FC(DevFault::SerialSilent)), 0);
    sleepMs(150);
    dm.logicTick();                                            // 全程无帧超时 → 边沿一次
    EXPECT_EQ(rec.fault(FC(DevFault::SerialSilent)), 1);
    dm.logicTick();                                            // 锁定不重复
    EXPECT_EQ(rec.fault(FC(DevFault::SerialSilent)), 1);

    kit.raw("G03 S1");                                         // 恢复帧（任意完整帧清锚）
    dm.logicTick();
    EXPECT_EQ(rec.fault(FC(DevFault::SerialSilent)), 1);
    sleepMs(150);
    dm.logicTick();                                            // 再超时 → 证明锚已清
    EXPECT_EQ(rec.fault(FC(DevFault::SerialSilent)), 2);
}

// —— F3（#3）：温度爆表 → 0x0803 边沿一次；持续超限不重复；回落清锚后再触发 ——
TEST(DeviceManager, F3_TempOverMaxEdge) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    Kit kit;
    DeviceConfig cfg = makeCfg();                              // tempMaxC 默认 60
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    kit.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    kit.temp(61.0);                                            // 爆表 → 边沿一次
    EXPECT_EQ(rec.fault(FC(DevFault::TempOverMax)), 1);
    kit.temp(61.5);                                            // 仍超限：锁定制不重复
    EXPECT_EQ(rec.fault(FC(DevFault::TempOverMax)), 1);
    kit.temp(55.0);                                            // 回落清锚
    kit.temp(61.0);                                            // 再爆 → 再触发
    EXPECT_EQ(rec.fault(FC(DevFault::TempOverMax)), 2);
}

// —— F4（#4）：温度乱跳绝对差值判据（批4：|Δ|>tempSpikeAbsC 默认 10℃/帧——
//      去上报周期敏感）→ 0x0804 边沿一次；差值回落清锚后可再触发 ——
TEST(DeviceManager, F4_TempSpikeAbsoluteDeltaEdge) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    Kit kit;
    DeviceConfig cfg = makeCfg();                              // tempSpikeAbsC 默认 10℃
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                      [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    kit.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    kit.temp(25.0);                                            // 基线帧（无前帧无警）
    EXPECT_EQ(rec.fault(FC(DevFault::TempSpike)), 0);
    kit.temp(30.0);                                            // |Δ5|<10：不报（旧速率判据会误报）
    EXPECT_EQ(rec.fault(FC(DevFault::TempSpike)), 0);
    kit.temp(41.5);                                            // |Δ11.5|>10 → 边沿一次
    EXPECT_EQ(rec.fault(FC(DevFault::TempSpike)), 1);
    kit.temp(41.5);                                            // 差值回落（Δ=0）→ 清锚
    EXPECT_EQ(rec.fault(FC(DevFault::TempSpike)), 1);
    kit.temp(55.0);                                            // |Δ13.5|>10 → 再触发
    EXPECT_EQ(rec.fault(FC(DevFault::TempSpike)), 2);
}

// —— F5（#5）：预热超时 → done(false) + 0x0805 恰一次（T4 基础上断言码）——
TEST(DeviceManager, F5_WarmupTimeoutFault) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    cfg.warmup.timeoutMs = 200;
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    int cbCount = 0;
    dm.startWarmup(42, [&](bool) { ++cbCount; });
    for (int i = 0; i < 60 && cbCount == 0; ++i) {
        sleepMs(25);
        dm.logicTick();
    }
    EXPECT_EQ(cbCount, 1);
    EXPECT_EQ(rec.fault(FC(DevFault::WarmupTimeout)), 1);      // D-T13：超时补 Fault
    EXPECT_EQ(mock.count("N13 S0"), 0);                        // 只报不停加热
}

// —— F6（#6）：手势洪峰挤爆 G01 环 → gestureRingDropped 增长 → 0x0806 一次；
//      无增量不重复 ——
TEST(DeviceManager, F6_GestureRingOverflowFault) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    Kit kit;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    kit.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    for (int i = 0; i < 100; ++i) kit.raw("G01 U1;");          // 100 手势帧 > 环容 63 → 丢新
    dm.logicTick();                                            // 泵消化 63 + 巡检报溢
    EXPECT_GE(rec.fault(FC(DevFault::KeyRingOverflow)), 1);
    dm.logicTick();                                            // 增量 0 → 不再报
    EXPECT_EQ(rec.fault(FC(DevFault::KeyRingOverflow)), 1);
}
