// ============================================================================
// test_device_manager.cpp — DeviceManager 门面集成测（D-T12b T1–T14 + D-T13 F1–F7）
//
// 全链真件（MCUDriver/KeyManager/KeySemantics/MenuLogic/ParamStore/Warmup/
// ModeController 全真配），假件仅两处边界：
//   - MockMcu：writeOverride 记下行帧 + 可配置自动 ACK 回执（收到 "$Nxx..seq..;"
//     解析 seq 回 "$A<seq>" 帧，经 DeviceManager::testInjectRaw 回灌）；
//   - FakeCamera：IScannerCamera 全接口空壳，isOpen 可拨（掉线模拟）。
// manualTick=true：不起逻辑线程，logicTick() 手动驱动；KeyManager/Warmup 时基
// 用真实系统钟（手势静默窗/预热窗以小阈值+毫秒级 sleep 换确定论）。
// 用例语义 = 08 设计方案 §7 集成行（T1–T12）+ T13/T14（并发冒烟/标定组链）+
// F1–F7（§6.2 故障 8 码：掉线边沿/心跳/温度双警/预热超时/K 环溢/seq 跳变）。
// ============================================================================

#include <gtest/gtest.h>

#include "modules/08_devicemgmt/DeviceManager.h"
#include "modules/08_devicemgmt/serial/FrameCodec.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace Scanner::device;
using FCodec = Scanner::device::serial::FrameCodec;
using Scanner::Event;
using Scanner::EventType;
using Scanner::Result;

namespace {

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
    // UserDefined 按 param1 计数（menuSelect ③/④ 出口——Important #4 去污染后）
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

// —— 假相机：全接口空壳 + isOpen 可控（T8 掉线模拟）——
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

// —— 假 MCU：记全部下行帧 + 可配置自动 ACK（noAck 前缀命中的命令不回执）——
struct MockMcu {
    DeviceManager* dm = nullptr;                  // ACK 回灌目标（open 后指向当前门面）
    std::vector<std::string> frames;
    std::vector<std::string> noAck;               // 不 ACK 的载荷前缀（如 "N10"）
    FCodec enc{FCodec::Version::V3};

    bool write(const std::string& f) {
        frames.push_back(f);
        if (!dm || f.empty() || f.front() != '$' || f.back() != ';') return true;  // v2 裸帧无 ACK
        const std::string body = f.substr(1, f.size() - 2);                        // payload+seq+crc
        if (body.size() < 6) return true;
        const std::string payload = body.substr(0, body.size() - 6);
        const std::string seqHex = body.substr(body.size() - 6, 2);
        for (const auto& p : noAck)
            if (payload.rfind(p, 0) == 0) return true;                             // 命中不回执
        dm->testInjectRaw(enc.encode("A" + seqHex, 0));                            // 回执 ACK
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
DeviceConfig makeCfg(FCodec::Version v = FCodec::Version::V3) {
    DeviceConfig c;
    c.serialPort = "COM_TEST";
    c.baud = 115200;
    c.protocol = v;
    c.ackTimeoutMs = 100;
    c.keys = GestureThresholds{10, 60, 60, 150};              // 消抖/短按/双击窗/长按
    c.warmup = WarmupConfig{100, 0.1, 2.0, 3000};             // 稳定窗 100ms
    c.manualTick = true;
    return c;
}

// —— 按键/温度注入工具（v3 帧经 testInjectRaw；手势经 MCU 时刻域合成）——
struct Kit {
    FCodec enc{FCodec::Version::V3};
    DeviceManager* dm = nullptr;
    uint16_t seq = 16;
    uint32_t mcu = 100;

    void raw(const std::string& payload) { dm->testInjectRaw(enc.encode(payload, seq++)); }
    void ev(char k, bool pressed, uint32_t t) {
        raw(std::string{'K', k, static_cast<char>(pressed ? '1' : '0'), ','} + std::to_string(t));
    }
    // 短按：按下→30ms 松开→静默窗到期（tick 判 S）
    void shortPress(char k) {
        ev(k, true, mcu);
        ev(k, false, mcu + 30);
        mcu += 1000;
        dm->logicTick();
        sleepMs(90);
        dm->logicTick();
        dm->logicTick();                                      // 追一拍消化命令 ACK
    }
    // 双击：两对按压松开（事件驱动判 D）
    void doublePress(char k) {
        ev(k, true, mcu);
        ev(k, false, mcu + 30);
        ev(k, true, mcu + 60);
        ev(k, false, mcu + 90);
        mcu += 1000;
        dm->logicTick();
        dm->logicTick();
    }
    // 长按：仅按下，holdMs 到期（tick 判 H）
    void holdPress(char k) {
        ev(k, true, mcu);
        mcu += 1000;
        dm->logicTick();
        sleepMs(190);
        dm->logicTick();
        dm->logicTick();
    }
    void temp(double c) {
        raw("T" + std::to_string(c));
        dm->logicTick();
    }
};

} // namespace

// —— T1：相机打开失败 → open fail + 倒序关闭无崩 + Fault 事件（MCU 未开无自检帧）——
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
    EXPECT_TRUE(mock.frames.empty());                          // MCU 未开：连 N12 Z1 都没发
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
    EXPECT_EQ(mock.frames.size(), 1u);                        // 仅 open 的 N12Z1
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
    dm.logicTick();                                            // 编队任务落地（N14T42 下发）
    EXPECT_EQ(mock.count("N14T42"), 1);                        // 加热命令已发

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
    EXPECT_EQ(mock.count("N14T42"), 1);
    EXPECT_EQ(mock.count("N14T0"), 0);                         // 超时不停加热
}

// —— T5：中键短按启停（N11 H1/H0 按黑板）+ 直调幂等（连按同值不乱）——
//      A-T17 修复后启采集=命令组 [N10(账本)→N11H1]：幂等断言同时覆盖 N10 ——
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

    kit.shortPress('M');                                       // 主层中键短按 → 启采集（组：N10→N11H1）
    EXPECT_EQ(mock.count("N11H1"), 1);
    EXPECT_EQ(mock.count("N10H60B80T1V1L120"), 1);             // 每次启采集先 N10（账本默认全参）
    EXPECT_TRUE(dm.isCapturing());
    kit.shortPress('M');                                       // 再按 → 停采集（单发 N11H0）
    EXPECT_EQ(mock.count("N11H0"), 1);
    EXPECT_FALSE(dm.isCapturing());

    dm.startCapture();                                         // 直调重复启：幂等无新帧
    dm.logicTick();
    EXPECT_EQ(mock.count("N11H1"), 2);                         // 组链一拍内完成（ACK 泵链推进）
    EXPECT_EQ(mock.count("N10H60B80T1V1L120"), 2);
    EXPECT_TRUE(dm.isCapturing());
    dm.startCapture();                                         // 采集已开：幂等无新 N10/N11
    dm.logicTick();
    EXPECT_EQ(mock.count("N11H1"), 2);
    EXPECT_EQ(mock.count("N10H60B80T1V1L120"), 2);
    dm.stopCapture();                                          // 直调重复停：幂等无新帧
    dm.logicTick();
    dm.stopCapture();
    dm.logicTick();
    EXPECT_EQ(mock.count("N11H0"), 2);
    EXPECT_FALSE(dm.isCapturing());
}

// —— T5b（A-T17 N10 断链修复钉死）：setParam 改账后 startCapture →
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
    EXPECT_EQ(mock.count("N10H90B80T1V1L120"), 1);              // N10 帧含 H90（账本值）
    EXPECT_EQ(mock.count("N11H1"), 1);
    EXPECT_TRUE(dm.isCapturing());
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
    kit.shortPress('U');                                       // 上键短按 L1：进菜单（cursor 复位①）
    EXPECT_EQ(st().layer, 2);
    EXPECT_EQ(st().cursor, 1);
    for (int i = 0; i < 4; ++i) kit.shortPress('R');           // 右键短按×4：1→2→3→4→1 环绕
    EXPECT_EQ(st().cursor, 1);
    kit.shortPress('L');                                       // 左键短按：1→4 环绕
    EXPECT_EQ(st().cursor, 4);
    kit.doublePress('M');                                      // 中键双击：模式光标 3→1→2→3
    EXPECT_EQ(st().modeCursor, 1);
    kit.doublePress('M');
    EXPECT_EQ(st().modeCursor, 2);
    kit.doublePress('M');
    EXPECT_EQ(st().modeCursor, 3);
    const int post0 = rec.userParam(4);
    kit.shortPress('M');                                       // 中键短按 L2 选中④：派后处理工作流（UserDefined p1=4）
    EXPECT_EQ(rec.userParam(4), post0 + 1);
    kit.shortPress('U');                                       // 上键短按 L2：退菜单
    EXPECT_EQ(st().layer, 1);

    kit.doublePress('U');                                      // 上键双击：None→View
    EXPECT_EQ(st().adjustCtx, MenuState::AdjustCtx::View);
    kit.shortPress('R');                                       // View 上下文：暂仅日志（曝光不动）
    const double base = dm.getParam("exposure").value;
    kit.doublePress('U');                                      // View→Brightness
    EXPECT_EQ(st().adjustCtx, MenuState::AdjustCtx::Brightness);
    kit.shortPress('R');                                       // 右键短按：曝光 +1ms（相机直设）
    EXPECT_DOUBLE_EQ(dm.getParam("exposure").value, base + 1.0);
    kit.shortPress('L');                                       // 左键短按：曝光 -1ms
    EXPECT_DOUBLE_EQ(dm.getParam("exposure").value, base);
    kit.doublePress('U');                                      // Brightness→None
    EXPECT_EQ(st().adjustCtx, MenuState::AdjustCtx::None);

    kit.shortPress('L');                                       // 主层无上下文左右：无效丢弃
    EXPECT_EQ(st().layer, 1);
    kit.doublePress('L');                                      // 双击/长按预留/无效手势全丢弃不崩
    kit.doublePress('R');
    kit.holdPress('U');
    kit.holdPress('M');
    kit.holdPress('L');
    kit.holdPress('R');
    EXPECT_EQ(st().layer, 1);
    // 快照一致性：末拍刷新后 menuState()（互斥快照口）= 逻辑线程账本状态
    EXPECT_EQ(dm.menuState().layer, 1);
    EXPECT_EQ(dm.menuState().adjustCtx, MenuState::AdjustCtx::None);
}

// —— T7：按键洪峰 100 帧 → 环有效容量 63（SpscRing<64> 满判 tail+1==head）收敛
//      （满丢新）→ ≥60 原始事件被消化（31 对完整手势）、无崩溃 ——
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
    mock.noAck = {"N11"};                                      // 关 ACK：每次启采集都发 H1（计消化数）
    ASSERT_TRUE(dm.open().success);

    uint32_t t = 100;
    for (int i = 0; i < 50; ++i) {                             // 50 对按下/松开 = 100 个 K 帧
        kit.ev('M', true, t);
        kit.ev('M', false, t + 30);
        t += 1000;
    }
    dm.logicTick();                                            // 环容量 64：仅前 64 事件入环
    sleepMs(90);                                               // 末对静默窗到期
    dm.logicTick();                                            // 手势 drain → 各组手势派发拍同步发 N10
    dm.logicTick();                                            // A-T17 组链：ACK 泵消化 → 链发 N11H1
    // 每消化一对手势产生一组 [N10→N11H1]：N10 在派发拍同步落帧（≥30 ⇔ ≥60 原始
    // 事件处理）；N11H1 经 ACK 泵链推进——洪峰下 CommandChannel 挂表容量有限，
    // 部分 N10 在链前被逐出判超时（容量策略「满：最旧先判超时」），只证链路活
    // （≥1），且全程无崩溃
    EXPECT_GE(mock.count("N10"), 30);
    EXPECT_GE(mock.count("N11H1"), 1);
    EXPECT_FALSE(dm.isCapturing());
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
    EXPECT_EQ(mock.count("N11H1"), 1);
    EXPECT_EQ(mock.count("N11H0"), 0);                         // 无自主停采
    EXPECT_FALSE(dm.isDeviceReady());
}

// —— T9：v2→close→v3 开关切换重连（v2 匿名按键丢、v3 手势链活）——
TEST(DeviceManager, T9_V2V3ProtocolSwitchReopen) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    Kit kit;
    auto write = [&](const std::string& f) { return mock.write(f); };

    {
        DeviceConfig v2 = makeCfg(FCodec::Version::V2);
        DeviceManager dm(v2, gateOk, &bus, nullptr, write);
        mock.dm = &dm;
        ASSERT_TRUE(dm.open().success);
        dm.testInjectRaw("K1;");                               // v2 匿名按键：构不出 RawKeyEvent → 丢
        dm.logicTick();
        EXPECT_EQ(dm.menuState().layer, 1);                    // 无任何手势副作用
        EXPECT_EQ(mock.count("N11"), 0);
    }                                                          // close（析构）

    DeviceConfig v3 = makeCfg(FCodec::Version::V3);
    DeviceManager dm(v3, gateOk, &bus, nullptr, write);
    mock.dm = &dm;
    kit.dm = &dm;
    ASSERT_TRUE(dm.open().success);
    kit.shortPress('M');                                       // v3 手势链正常
    EXPECT_EQ(mock.count("N11H1"), 1);
    EXPECT_TRUE(dm.isCapturing());
}

// —— T10：ACK 丢失 → 1+3 重传后 Fault；期间 logicTick 非阻塞可推进 ——
TEST(DeviceManager, T10_AckLossRetransmitNonBlocking) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    cfg.ackTimeoutMs = 30;
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    mock.noAck = {"N11"};                                      // 模拟 ACK 石沉大海
    ASSERT_TRUE(dm.open().success);

    dm.startCapture();
    dm.logicTick();                                            // 编队任务落地（N11H1 首发）
    EXPECT_EQ(mock.count("N11H1"), 1);
    const auto t0 = std::chrono::steady_clock::now();          // 非阻塞证明：连 10 拍立即返回
    for (int i = 0; i < 10; ++i) dm.logicTick();
    const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    EXPECT_LT(dt, 500);
    EXPECT_EQ(mock.count("N11H1"), 1);                         // 无时间推进 → 无重传

    for (int i = 0; i < 50 && mock.count("N11H1") < 4; ++i) {  // 重传×3 + 3 败收口
        sleepMs(5);
        dm.logicTick();
    }
    EXPECT_EQ(mock.count("N11H1"), 4);
    EXPECT_GE(rec.count(EventType::FaultOccurred), 1);
    EXPECT_FALSE(dm.isCapturing());
}

// —— T11：v2 降级全链——启停立返 ok「未确认」无重传、被动收温、断流 1.2s N15 兜底 ——
TEST(DeviceManager, T11_V2DegradedFullChain) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg(FCodec::Version::V2);
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    dm.startCapture();                                         // v2：编队执行 send 内立即回调 ok「未确认」
    dm.logicTick();
    EXPECT_TRUE(dm.isCapturing());
    EXPECT_EQ(mock.count("N11H1"), 1);
    for (int i = 0; i < 10; ++i) {
        sleepMs(10);
        dm.logicTick();
    }
    EXPECT_EQ(mock.count("N11H1"), 1);                         // 无 ACK 不重传不判败
    EXPECT_EQ(rec.count(EventType::FaultOccurred), 0);

    dm.testInjectRaw("T25.3;");                                // v2 被动收现状 T 帧（单路）
    dm.logicTick();
    EXPECT_EQ(dm.getLastTemperatures().channels, 1);
    EXPECT_DOUBLE_EQ(dm.getLastTemperatures().celsius[0], 25.3);

    for (int i = 0; i < 70; ++i) {                             // 断流 ~1.4s → 兜底查询恰一次
        sleepMs(20);
        dm.logicTick();
    }
    EXPECT_EQ(mock.count("N15V2"), 1);
}

// —— T12：enterScan 命令组中段 3 败 → 不擦板+Fault；对照全 ACK → 擦板+采集开 ——
TEST(DeviceManager, T12_GroupMidFailVersusFullAckCommit) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    cfg.ackTimeoutMs = 50;
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    mock.noAck = {"N10"};                                      // N10 ACK 丢失 → 组中段 3 败
    ASSERT_TRUE(dm.open().success);

    dm.toIdle();                                               // N13E1 全 ACK → 落板待机
    dm.logicTick();
    ASSERT_EQ(dm.mode(), DeviceMode::Idle);
    EXPECT_EQ(mock.count("N13E1"), 1);

    dm.enterScan();                                            // 组：N13E0→N10(3败)→N11 短路
    for (int i = 0; i < 80 && mock.count("N10") < 4; ++i) {
        sleepMs(5);
        dm.logicTick();
    }
    EXPECT_EQ(mock.count("N10"), 4);                           // N10 首发+重传×3
    EXPECT_EQ(mock.count("N13E0"), 1);
    EXPECT_EQ(mock.count("N11H1"), 0);                         // 组短路：N11 未发
    EXPECT_EQ(dm.mode(), DeviceMode::Idle);                    // 黑板不落 Scanning
    EXPECT_FALSE(dm.isCapturing());
    EXPECT_GE(rec.count(EventType::FaultOccurred), 1);
    EXPECT_EQ(rec.count(EventType::StateChanged), 0);          // toIdle=same-mode 落板不广播（D-T9 口径）

    mock.noAck.clear();                                        // 对照：全 ACK 路径
    dm.enterScan();
    for (int i = 0; i < 10; ++i) dm.logicTick();
    EXPECT_EQ(mock.count("N10H60B80T1V1L120"), 5);             // N10 全参自 ParamStore 账本（前段 4 + 本段 1）
    EXPECT_EQ(mock.count("N11H1"), 1);
    EXPECT_EQ(dm.mode(), DeviceMode::Scanning);
    EXPECT_TRUE(dm.isCapturing());
    EXPECT_EQ(rec.count(EventType::StateChanged), 1);          // commit(Scanning) 落板广播恰一次
}

// —— T13（Critical #1 回归）：双线程真并发冒烟——manualTick=false 起真逻辑线程，
//      另一线程连发 50 次 setParam+startCapture/stopCapture 交替（+并发 getParam
//      快照读），2s 后 close 停线程清队——无死锁无崩溃（互踩冒烟；TSAN 级
//      确定性验证归 T18 收口）——
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
    EXPECT_GE(mock.count("N11H1") + mock.count("N11H0"), 1);   // 任务确有落地
    EXPECT_GE(dm.getParam("bgLight").value, 0.0);              // 快照口仍可读
}

// —— T14（Important #3 补缺）：enterCalibration 组链——N16 3 败→不擦板+Fault；
//      全 ACK→commit Calibrating+StateChanged 恰一次（MCU open 失败回滚分支由
//      T1 相机败回滚用例+代码审查双覆盖——writeOverride 测试模式 open 恒成功）——
TEST(DeviceManager, T14_EnterCalibrationGroupChain) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    cfg.ackTimeoutMs = 50;
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    mock.noAck = {"N16"};                                      // N16 ACK 丢失 → 组 3 败
    ASSERT_TRUE(dm.open().success);

    EXPECT_TRUE(dm.enterCalibration().success);                // 门禁过（编队执行）
    for (int i = 0; i < 80 && mock.count("N16B1") < 4; ++i) {  // 首发+重传×3
        sleepMs(5);
        dm.logicTick();
    }
    EXPECT_EQ(mock.count("N16B1"), 4);
    EXPECT_EQ(dm.mode(), DeviceMode::Idle);                    // 黑板不落 Calibrating
    EXPECT_EQ(rec.count(EventType::StateChanged), 0);
    EXPECT_GE(rec.count(EventType::FaultOccurred), 1);

    mock.noAck.clear();                                        // 对照：全 ACK 路径
    EXPECT_TRUE(dm.enterCalibration().success);
    for (int i = 0; i < 10; ++i) dm.logicTick();
    EXPECT_EQ(mock.count("N16B1"), 5);                         // 前段 4 + 本段 1
    EXPECT_EQ(dm.mode(), DeviceMode::Calibrating);             // 组成功才擦板
    EXPECT_EQ(rec.count(EventType::StateChanged), 1);          // 落板广播恰一次
}

// ============================================================================
// D-T13：故障 8 码接线（设计方案 §6.2 十类事故 → 8 码；边沿纪律=恢复清锚）
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

// —— F2（#2≡#10）：心跳超时 → 0x0802 边沿一次；恢复帧清锚后可再触发 ——
TEST(DeviceManager, F2_HeartbeatTimeoutEdgeAndRecover) {
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
    ASSERT_TRUE(dm.open().success);                            // N12Z1 ACK 回灌 → lastRx>0

    dm.logicTick();                                            // 距末帧 <100ms：无声警
    EXPECT_EQ(rec.fault(FC(DevFault::SerialSilent)), 0);
    sleepMs(150);
    dm.logicTick();                                            // 超时 → 边沿一次
    EXPECT_EQ(rec.fault(FC(DevFault::SerialSilent)), 1);
    dm.logicTick();                                            // 锁定不重复
    EXPECT_EQ(rec.fault(FC(DevFault::SerialSilent)), 1);

    kit.raw("T20.0");                                          // 恢复帧（任意有效帧清锚）
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

// —— F4（#4）：温度乱跳 → 0x0804 边沿一次；平稳帧清锚后可再触发 ——
TEST(DeviceManager, F4_TempSpikeEdge) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    Kit kit;
    DeviceConfig cfg = makeCfg();                              // tempSpikeC 默认 2.0℃/s
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    kit.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    kit.temp(25.0);                                            // 基线帧（无前帧无警）
    EXPECT_EQ(rec.fault(FC(DevFault::TempSpike)), 0);
    kit.temp(30.0);                                            // 相邻帧 |Δ5|/<1s → 速率远超 2℃/s
    EXPECT_EQ(rec.fault(FC(DevFault::TempSpike)), 1);
    kit.temp(30.0);                                            // 平稳帧（Δ=0）→ 清锚
    EXPECT_EQ(rec.fault(FC(DevFault::TempSpike)), 1);
    kit.temp(36.0);                                            // 再跳 → 再触发
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
    EXPECT_EQ(mock.count("N14T0"), 0);                         // 只报不停加热
}

// —— F6（#6）：按键洪峰挤爆 K 环 → keyDrop 增长 → 0x0806 一次；无增量不重复 ——
TEST(DeviceManager, F6_KeyRingOverflowFault) {
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

    uint32_t t = 100;
    for (int i = 0; i < 50; ++i) {                             // 100 K 帧 > 环容 63 → 丢新
        kit.ev('M', true, t);
        kit.ev('M', false, t + 30);
        t += 1000;
    }
    dm.logicTick();                                            // 泵消化 + 巡检报溢
    EXPECT_GE(rec.fault(FC(DevFault::KeyRingOverflow)), 1);
    dm.logicTick();                                            // 增量 0 → 不再报
    EXPECT_EQ(rec.fault(FC(DevFault::KeyRingOverflow)), 1);
}

// —— F7（#9）：T 帧 seq 跳变 → 对账计数增长 → 0x0808 一次；连续 seq 不重复 ——
TEST(DeviceManager, F7_SeqGapFault) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    Kit kit;
    DeviceConfig cfg = makeCfg();
    cfg.seqGapWarn = 1;                                        // 测试注入：1 跳即警
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    kit.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    dm.testInjectRaw(kit.enc.encode("T25.0", 10));             // 对账基线
    dm.logicTick();
    EXPECT_EQ(rec.fault(FC(DevFault::SeqGap)), 0);
    dm.testInjectRaw(kit.enc.encode("T25.0", 16));             // v3 下 seq 10→16 跳变
    dm.logicTick();
    EXPECT_EQ(rec.fault(FC(DevFault::SeqGap)), 1);
    dm.testInjectRaw(kit.enc.encode("T25.0", 17));             // 连续 → 无增量不重复
    dm.logicTick();
    EXPECT_EQ(rec.fault(FC(DevFault::SeqGap)), 1);
}
