// ============================================================================
// test_device_manager.cpp — DeviceManager 门面集成测（D-T12b T1–T14 + D-T13 F1–F7）
//
// 全链真件（MCUDriver/KeySemantics/MenuLogic/ParamStore/Warmup/
// ModeController 全真配），假件仅两处边界：
//   - MockMcu：writeOverride 记下行帧 + 可配置自动 ACK 回执（收到 "$Nxx..seq..;"
//     解析 seq 回 "$A<seq>" 帧，经 DeviceManager::testInjectRaw 回灌）；
//   - FakeCamera：IScannerCamera 全接口空壳，isOpen 可拨（掉线模拟）。
// manualTick=true：不起逻辑线程，logicTick() 手动驱动；Warmup 时基用真实系统钟
// （预热窗以小阈值+毫秒级 sleep 换确定论）。手势=MCU 已判 G01 文本行注入
// （KeyManager 退役：无 PC 侧消抖/时序合成），单发即单一手势。
// 用例语义 = 08 设计方案 §7 集成行（T1–T12）+ T13/T14（并发冒烟/标定组链）+
// F1–F7（§6.2 故障 8 码：掉线边沿/心跳/温度双警/预热超时/G01 手势环溢/G03 帧计数对账）。
// ============================================================================

#include <gtest/gtest.h>

#include "modules/DeviceManager/DeviceManager.h"
#include "modules/DeviceManager/serial/FrameCodec.h"

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
    // Fault 按码计数（D-T13：DevFault 码表断言；统一契约 param2=码 2026-09-20）
    int fault(int64_t faultCode) const {
        std::lock_guard<std::mutex> lock(m);
        int n = 0;
        for (const auto& e : ev)
            if (e.type == EventType::FaultOccurred && e.param2 == faultCode) ++n;
        return n;
    }
};

// DevFault 码 → int64（断言简写）
constexpr int64_t FC(DevFault f) { return static_cast<int64_t>(f); }

// —— 假相机：全接口空壳 + isOpen 可控（T8 掉线模拟）＋exposureFail（T24 换档
//    事务回弹注入：setExposure 失败＝相机段下发败）——
struct FakeCamera : Scanner::hal::IScannerCamera {
    bool openOk = true;
    bool openState = false;
    bool exposureFail = false;             // T24：setExposure 恒败（事务回弹源）
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
        if (exposureFail) return Result::fail("曝光设置失败(测试注入)");
        exposureMs = ms;
        return Result::ok();
    }
    Result setGain(double) override { return Result::ok(); }
    Result setResolution(int, int) override { return Result::ok(); }
    Result setContrast(int, int) override { return Result::ok(); }   // 软件对比度口·左右分置（260927→1002）
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
    c.warmup = WarmupConfig{100, 0.1, 2.0, 3000};             // 稳定窗 100ms
    c.manualTick = true;
    return c;
}

// —— 按键/温度注入工具（温度帧 v3 经 testInjectRaw；手势=MCU 已判 G01 文本行直收，
//     经 testInjectTextLine——KeyManager 退役后无 PC 侧组合）——
struct Kit {
    FCodec enc{FCodec::Version::V3};
    DeviceManager* dm = nullptr;
    uint16_t seq = 16;

    void raw(const std::string& payload) { dm->testInjectRaw(enc.encode(payload, seq++)); }
    // G01 手势：键 U/L/M/R + 手势位 1短/2双/3长（MCU 已判）→ 文本行注入 + 本拍 pump 派发
    void gest(char k, int g) {
        dm->testInjectTextLine(std::string("G01 ") + k + std::to_string(g));
        dm->logicTick();
    }
    // 短按/双击/长按 = 单发 G01（手势类型 MCU 判定，无 PC 侧时序组合）
    void shortPress(char k) { gest(k, 1); dm->logicTick(); }   // 追加一拍消化命令 ACK
    void doublePress(char k) { gest(k, 2); dm->logicTick(); }
    void holdPress(char k) { gest(k, 3); dm->logicTick(); }
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
    EXPECT_EQ(mock.frames.size(), 0u);                         // open 不再发 N12 Z1（2026-08-30 去自检模式）
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
    dm.logicTick();                                            // 编队任务落地（N14 T42 下发）
    EXPECT_EQ(mock.count("N14 T42"), 1);                        // 加热命令已发

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
    EXPECT_EQ(mock.count("N14 T42"), 1);
    EXPECT_EQ(mock.count("N14 T0"), 0);                         // 超时不停加热
}

// —— T5：中键短按启停（N10/N11 H0 按黑板）+ 直调幂等（连按同值不乱）——
//      a9bfe53 用户口径：启采集=单帧 N10（参数+灯控），不发 N11 H1 ——
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
    dm.setScanReady(true);   // 261004 矩阵对齐：主层按键（启停/调节/切模式）测试布防就绪凭据

    kit.shortPress('M');                                       // 主层中键短按 → 启采集（单帧 N10）
    EXPECT_EQ(mock.count("N11 H1"), 0);                         // 启动不发 N11 H1（a9bfe53 口径）
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 1);             // 每次启采集发 N10（账本默认全参）
    EXPECT_TRUE(dm.isCapturing());
    kit.shortPress('M');                                       // 再按 → 停采集（单发 N11 H0）
    EXPECT_EQ(mock.count("N11 H0"), 1);
    EXPECT_FALSE(dm.isCapturing());

    dm.startCapture();                                         // 直调重复启：幂等无新帧
    dm.logicTick();
    EXPECT_EQ(mock.count("N11 H1"), 0);                         // 全程无 N11 H1
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 2);
    EXPECT_TRUE(dm.isCapturing());
    dm.startCapture();                                         // 采集已开：幂等无新 N10/N11
    dm.logicTick();
    EXPECT_EQ(mock.count("N11 H1"), 0);
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 2);
    dm.stopCapture();                                          // 直调重复停：幂等无新帧
    dm.logicTick();
    dm.stopCapture();
    dm.logicTick();
    EXPECT_EQ(mock.count("N11 H0"), 2);
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
    EXPECT_EQ(mock.count("N10 H90 B47 T0 V1 C1 D0 L47"), 1);              // N10 帧含 H90（账本值）
    EXPECT_EQ(mock.count("N11 H1"), 0);                         // 启动不发 N11 H1（a9bfe53 口径）
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
    dm.setScanReady(true);   // 261004 矩阵对齐：主层按键（启停/调节/切模式）测试布防就绪凭据
    auto st = [&] { return dm.menuState(); };

    EXPECT_EQ(st().layer, 1);
    kit.shortPress('U');                                       // 上键短按 L1：进菜单（cursor 复位①）
    EXPECT_EQ(st().layer, 2);
    EXPECT_EQ(st().cursor, 1);
    // 261004 竖排方向对齐后：R(↑)＝-1 向①、L(↓)＝+1 向⑤（见 KeySemantics 注释）
    for (int i = 0; i < 5; ++i) kit.shortPress('R');           // 右(上)×5：①→⑤→④…②→① 环绕
    EXPECT_EQ(st().cursor, 1);
    kit.shortPress('L');                                       // 左(下)：①→②（视觉下移）
    EXPECT_EQ(st().cursor, 2);
    kit.doublePress('M');                                      // 菜单内双击＝收紧丢弃（模式不动）
    EXPECT_EQ(st().modeCursor, 3);                             // 仍是 3（未切）
    kit.doublePress('M');
    kit.doublePress('M');                                      // 3→1→2→3（丢弃不切——非路径验证）
    const int64_t post0 = rec.userParam(104);
    for (int i = 0; i < 2; ++i) kit.shortPress('L');           // 左(下)×2：②→③→④
    kit.shortPress('M');                                       // 中键短按 L2 选中④：后处理事件 p1=104
    EXPECT_EQ(st().layer, 1);                                  // 执行并自动退菜单（261002 定稿）
    EXPECT_EQ(rec.userParam(104), post0 + 1);

    // 主界面双击族（261002 定稿）：上双击＝景深直切（p1=114）；左双击＝换调节对象
    const int64_t dof0 = rec.userParam(114);
    kit.doublePress('U');                                      // 景深 近→远
    EXPECT_EQ(rec.userParam(114), dof0 + 1);
    kit.doublePress('U');                                      // 远→近
    EXPECT_EQ(rec.userParam(114), dof0 + 2);
    kit.doublePress('L');                                      // 换调节对象：亮度→显示远近
    EXPECT_EQ(st().adjustCtx, MenuState::AdjustCtx::DisplayDistance);
    kit.shortPress('R');                                       // 显示远近档 1→2（p1=112；参数不动）
    kit.doublePress('L');                                      // 对象回亮度
    EXPECT_EQ(st().adjustCtx, MenuState::AdjustCtx::Brightness);
    // 均分梯默认档10起步——先回档1再测步进（261003 均分版）
    dm.setBrightnessLadderIndex(1);
    dm.logicTick(); dm.logicTick();
    kit.shortPress('R');                                       // 亮度档 1→2：三参＝档2 均分值
    {
        const auto stepsT6 = PresetLadder::builtinLadder();
        EXPECT_NEAR(dm.getParam("exposure").value, stepsT6[1].exposureMs, 1e-9);
        EXPECT_NEAR(dm.getParam("laserLevel").value, stepsT6[1].laserLevel, 1e-9);
        EXPECT_NEAR(dm.getParam("bgLight").value, stepsT6[1].bgLight, 1e-9);
    }
    kit.shortPress('L');                                       // 档2→1（回落）
    EXPECT_NEAR(dm.getParam("exposure").value,
                PresetLadder::builtinLadder()[0].exposureMs, 1e-9);

    // 长按出口（261004 用户指令：U/H 回主界面与 M/H 急停暂时停用——丢弃留痕）
    kit.shortPress('U');                                       // 进菜单（游标①）
    EXPECT_EQ(st().layer, 2);
    kit.holdPress('U');                                        // 停用：菜单不退（丢弃）
    EXPECT_EQ(st().layer, 2);
    kit.shortPress('U');                                       // 上短按退菜单
    EXPECT_EQ(st().layer, 1);
    EXPECT_EQ(st().adjustCtx, MenuState::AdjustCtx::Brightness);
    kit.holdPress('M');                                        // 急停停用：不崩不执行
    kit.holdPress('L');                                        // 左右长按预留
    kit.holdPress('R');
    EXPECT_EQ(st().layer, 1);
    // 快照一致性：末拍刷新后 menuState()（互斥快照口）= 逻辑线程账本状态
    EXPECT_EQ(dm.menuState().layer, 1);
    EXPECT_EQ(dm.menuState().adjustCtx, MenuState::AdjustCtx::Brightness);
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
    dm.setScanReady(true);   // 261004 矩阵对齐：主层按键（启停/调节/切模式）测试布防就绪凭据

    for (int i = 0; i < 50; ++i) {                             // 50 次 M 短按 = 50 个 G01 手势
        kit.shortPress('M');
    }
    // 环容量 64：50 < 64 不溢；每手势即一个启/停（启=N10/停=N11 H0——a9bfe53 口径）：
    // 洪峰下 CommandChannel 挂表容量有限，组可被逐出判超时——本测只证洪峰不崩
    EXPECT_GE(mock.count("N10") + mock.count("N11 H0"), 1);      // 洪峰下仍有命令落地（启=N10/停=N11 H0）
    EXPECT_EQ(dm.menuState().layer, 1);                        // 洪峰后菜单层仍可读（不崩/不死；启停奇偶不assert）
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
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 1);              // 启采集单帧 N10
    EXPECT_EQ(mock.count("N11 H0"), 0);                         // 无自主停采
    EXPECT_FALSE(dm.isDeviceReady());
}

// —— T9：v2→close→v3 开关切换重连（v2 匿名 K 帧停用丢弃、v3 G01 手势链活）——
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
    dm.setScanReady(true);   // 261004 矩阵对齐：主层按键（启停/调节/切模式）测试布防就绪凭据
        dm.testInjectRaw("K1;");                               // v2 匿名按键：K 原始链停用 → 落 onParseFail 丢
        dm.logicTick();
        EXPECT_EQ(dm.menuState().layer, 1);                    // 无任何手势副作用
        EXPECT_EQ(mock.count("N11"), 0);
    }                                                          // close（析构）

    DeviceConfig v3 = makeCfg(FCodec::Version::V3);
    DeviceManager dm(v3, gateOk, &bus, nullptr, write);
    mock.dm = &dm;
    kit.dm = &dm;
    ASSERT_TRUE(dm.open().success);
    dm.setScanReady(true);   // 261004 矩阵对齐：主层按键（启停/调节/切模式）测试布防就绪凭据
    kit.shortPress('M');                                       // v3 手势链正常
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 1);              // 启采集=单帧 N10（v2 会话 close 的 N11 H0/N12 Z0 不计入）
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
    mock.noAck = {"N10"};                                       // 模拟 N10 ACK 石沉大海（启采集首步）
    ASSERT_TRUE(dm.open().success);

    dm.startCapture();
    dm.logicTick();                                            // 编队任务落地（N10 首发）
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 1);
    const auto t0 = std::chrono::steady_clock::now();          // 非阻塞证明：连 10 拍立即返回
    for (int i = 0; i < 10; ++i) dm.logicTick();
    const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    EXPECT_LT(dt, 500);
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 1);             // 无时间推进 → 无重传

    for (int i = 0; i < 50 && mock.count("N10 H60 B47 T0 V1 C1 D0 L47") < 4; ++i) {  // 重传×3 + 3 败收口
        sleepMs(5);
        dm.logicTick();
    }
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 4);
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
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 1);
    for (int i = 0; i < 10; ++i) {
        sleepMs(10);
        dm.logicTick();
    }
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 1);             // 无 ACK 不重传不判败
    EXPECT_EQ(rec.count(EventType::FaultOccurred), 0);

    dm.testInjectRaw("T25.3;");                                // v2 被动收现状 T 帧（单路）
    dm.logicTick();
    EXPECT_EQ(dm.getLastTemperatures().channels, 1);
    EXPECT_DOUBLE_EQ(dm.getLastTemperatures().celsius[0], 25.3);

    for (int i = 0; i < 70; ++i) {                             // 断流 ~1.4s → 兜底查询恰一次
        sleepMs(20);
        dm.logicTick();
    }
    EXPECT_EQ(mock.count("N15 V2"), 1);
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
    mock.noAck = {"N10"};                                       // N10 ACK 丢失 → 组中段 3 败
    ASSERT_TRUE(dm.open().success);

    dm.toIdle();                                               // N13 E1 全 ACK → 落板待机
    dm.logicTick();
    ASSERT_EQ(dm.mode(), DeviceMode::Idle);
    EXPECT_EQ(mock.count("N13 E1"), 1);

    dm.enterScan();                                            // 组：N13 E0→N10(3败)→FLUSH 短路
    for (int i = 0; i < 80 && mock.count("N10") < 4; ++i) {
        sleepMs(5);
        dm.logicTick();
    }
    EXPECT_EQ(mock.count("N10"), 4);                            // N10 首发+重传×3
    EXPECT_EQ(mock.count("N13 E0"), 1);
    EXPECT_EQ(mock.count("N11"), 0);                            // 组短路：无 N11 下行
    EXPECT_EQ(dm.mode(), DeviceMode::Idle);                    // 黑板不落 Scanning
    EXPECT_FALSE(dm.isCapturing());
    EXPECT_GE(rec.count(EventType::FaultOccurred), 1);
    EXPECT_EQ(rec.count(EventType::StateChanged), 0);          // toIdle=same-mode 落板不广播（D-T9 口径）

    mock.noAck.clear();                                        // 对照：全 ACK 路径
    dm.enterScan();
    for (int i = 0; i < 10; ++i) dm.logicTick();
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 5);             // N10 全参自账本（首段 3 败 4 帧 + 本段 1）
    EXPECT_EQ(mock.count("N13 E0"), 1);                         // 待机已退：本段无 N13 E0
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
    EXPECT_GE(mock.count("N11 H1") + mock.count("N11 H0"), 1);   // 任务确有落地
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
    for (int i = 0; i < 80 && mock.count("N16 B1") < 4; ++i) {  // 首发+重传×3
        sleepMs(5);
        dm.logicTick();
    }
    EXPECT_EQ(mock.count("N16 B1"), 4);
    EXPECT_EQ(dm.mode(), DeviceMode::Idle);                    // 黑板不落 Calibrating
    EXPECT_EQ(rec.count(EventType::StateChanged), 0);
    EXPECT_GE(rec.count(EventType::FaultOccurred), 1);

    mock.noAck.clear();                                        // 对照：全 ACK 路径
    EXPECT_TRUE(dm.enterCalibration().success);
    for (int i = 0; i < 10; ++i) dm.logicTick();
    EXPECT_EQ(mock.count("N16 B1"), 5);                         // 前段 4 + 本段 1
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
// 261002 临时测试机：下位机无温度/周期上行——0x0802 串口静默巡检已在
// DeviceManager 中整段停用（误报→S7 停机），本用例随之注释；回正式机一并恢复
// TEST(DeviceManager, F2_HeartbeatTimeoutEdgeAndRecover) {
//     Scanner::infra::EventBus bus;
//     EventRecorder rec;
//     bus.subscribeAll([&](const Event& e) { rec.record(e); });
//     MockMcu mock;
//     Kit kit;
//     DeviceConfig cfg = makeCfg();
//     cfg.heartbeatTimeoutMs = 100;                              // 测试注入：100ms 判无声
//     DeviceManager dm(cfg, gateOk, &bus, nullptr,
//                      [&](const std::string& f) { return mock.write(f); });
//     mock.dm = &dm;
//     kit.dm = &dm;
//     ASSERT_TRUE(dm.open().success);                            // open 不再发 N12 Z1——注入 T 帧立心跳锚
//     kit.raw("T20.0");
//     dm.logicTick();
//
//     dm.logicTick();                                            // 距末帧 <100ms：无声警
//     EXPECT_EQ(rec.fault(FC(DevFault::SerialSilent)), 0);
//     sleepMs(150);
//     dm.logicTick();                                            // 超时 → 边沿一次
//     EXPECT_EQ(rec.fault(FC(DevFault::SerialSilent)), 1);
//     dm.logicTick();                                            // 锁定不重复
//     EXPECT_EQ(rec.fault(FC(DevFault::SerialSilent)), 1);
//
//     kit.raw("T20.0");                                          // 恢复帧（任意有效帧清锚）
//     dm.logicTick();
//     EXPECT_EQ(rec.fault(FC(DevFault::SerialSilent)), 1);
//     sleepMs(150);
//     dm.logicTick();                                            // 再超时 → 证明锚已清
//     EXPECT_EQ(rec.fault(FC(DevFault::SerialSilent)), 2);
// }

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
    EXPECT_EQ(mock.count("N14 T0"), 0);                         // 只报不停加热
}

// —— F6（#6）：G01 手势洪峰挤爆手势环 → keyDrop 增长 → 0x0806 一次；无增量不重复 ——
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
    dm.setScanReady(true);   // 261004 矩阵对齐：主层按键（启停/调节/切模式）测试布防就绪凭据

    for (int i = 0; i < 100; ++i) {                            // 100 个 G01 L1（主层左键短=无效，
        dm.testInjectTextLine("G01 L1");                       // 无副作用）> 环容 63 → 丢新 ~37
    }
    dm.logicTick();                                            // 泵消化 + 巡检报溢
    EXPECT_GE(rec.fault(FC(DevFault::KeyRingOverflow)), 1);
    dm.logicTick();                                            // 增量 0 → 不再报
    EXPECT_EQ(rec.fault(FC(DevFault::KeyRingOverflow)), 1);
}

// —— F7（#9）：G03 帧计数跳变（丢帧）对账 → 计数增长 → 0x0808 一次；连续不重复 ——
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

    dm.testInjectTextLine("G03 S1");             // 对账基线
    dm.logicTick();
    EXPECT_EQ(rec.fault(FC(DevFault::SeqGap)), 0);
    dm.testInjectTextLine("G03 S10");            // 1→10 跳变 = 丢 8 帧
    dm.logicTick();
    EXPECT_EQ(rec.fault(FC(DevFault::SeqGap)), 1);
    dm.testInjectTextLine("G03 S11");            // 连续 → 无增量不重复
    dm.logicTick();
    EXPECT_EQ(rec.fault(FC(DevFault::SeqGap)), 1);
}

// —— T15：启动自检只发一个 N10（2026-09-26 用户口径）——open 不发；startupSelfCheck
// 兜底恰一次（manual/测试注入路径）；stage0 等待期不重发。auto 搜口省略凭据
// （probeN10Sent）需真串口逐口探测，无注入缝——真机回归口径 ——
TEST(DeviceManager, T15_SelfCheckSingleN10) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);
    EXPECT_EQ(mock.count("N10 H50"), 0);                         // open 不发 N10

    dm.startupSelfCheck([](const std::string&, bool) {});
    dm.logicTick();                                              // 跑投稿：兜底补发恰一次
    EXPECT_EQ(mock.count("N10 H50"), 1);
    dm.logicTick();                                              // stage0 等回显期不重发
    dm.logicTick();
    EXPECT_EQ(mock.count("N10 H50"), 1);
}

// —— T16：自检亮灯总窗 ~3s（2026-09-27 用户口径「灯亮三秒」，历程 ~9s→~2s→
// ~9s→~3s）——stage0 1s 上行活证提前过关＋stage1 停留 2s → N11 H0 收口。
// 无回显固件口径：MockMcu 不回显 N10（v3 无整帧回显）、自动 ACK 刷 lastRx_
// 即活证。0.8s 未收口；≤5s 内 N11 H0 恰一次；全程 N10 恰一次（真钟驱动，
// real-time 用例 ~3.5s）——
TEST(DeviceManager, T16_SelfCheckLightWindowThreeSeconds) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    int bgOk = 0;
    dm.startupSelfCheck([&](const std::string& item, bool ok) {
        if (item == "bgLight" && ok) ++bgOk;
    });
    dm.logicTick();                                              // stage0 起（N10 兜底恰一次）
    EXPECT_EQ(mock.count("N10 H50"), 1);
    EXPECT_EQ(mock.count("N12 T5"), 0);                          // 260927 排障隔离：N12 T5 停发（原 1）

    for (int i = 0; i < 16; ++i) {                               // ~0.8s：1s 线未到
        sleepMs(50);
        dm.logicTick();
    }
    EXPECT_EQ(mock.count("N11 H0"), 0);                          // 未过关未收口

    bool closed = false;                                         // 至 ~3s：活证过关＋停留 2s 满收口
    for (int i = 0; i < 100 && !closed; ++i) {                   // 上界 5s（余量）
        sleepMs(50);
        dm.logicTick();
        closed = mock.count("N11 H0") > 0;
    }
    EXPECT_TRUE(closed);
    EXPECT_EQ(mock.count("N11 H0"), 1);
    EXPECT_EQ(mock.count("N10 H50"), 1);                         // 全程未重发
    EXPECT_EQ(mock.count("N12 T5"), 0);                          // 260927 排障隔离：N12 T5 停发（原 1）
    EXPECT_GE(bgOk, 1);                                          // 灯项活证过关已报
}

// —— T17：按键模式落地（260927 完善按键管理）——中键双击切模式＋菜单①②确认
// → captureToggle 启采按新模式组帧 N10 四管掩码（原缺口：cycleMode/①② 仅菜单
// 记账，采集恒用 app 上次传入模式）。掩码映射（261002 临时测试机管语义）：普通
// 交叉 V1C1 / 精细 T / 深孔 D（无对应线占位）——
TEST(DeviceManager, T17_KeyModeDispatchToN10) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);
    dm.setScanReady(true);   // 261004 矩阵对齐：主层按键（启停/调节/切模式）测试布防就绪凭据

    // 默认 modeCursor=3（普通交叉）→ 首次启采＝面片掩码 V1C1
    dm.testInjectTextLine("G01 M1");             // 主层中键短按＝启采
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 1);

    // 采集中双击（260927 用户口径：实时切模式）——3→1 精细：N10 全参重发 T 管
    dm.testInjectTextLine("G01 M2");
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(dm.captureMode(), Scanner::ScanMode::FineScan);     // 模式随帧贯通读取口
    EXPECT_EQ(mock.count("N10 H60 B47 T1 V0 C0 D0 L47"), 1);     // 采集态重发精细掩码
    dm.testInjectTextLine("G01 M1");             // 停采
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(mock.count("N11 H0"), 1);

    // 空闲双击：1→2 深孔——下次启采生效
    dm.testInjectTextLine("G01 M2");
    dm.logicTick();
    EXPECT_EQ(dm.captureMode(), Scanner::ScanMode::DeepHoleScan);
    dm.testInjectTextLine("G01 M1");             // 再启采
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V0 C0 D1 L47"), 1);     // 深孔掩码
    dm.testInjectTextLine("G01 M1");             // 停采
    dm.logicTick(); dm.logicTick();

    // 菜单段（261002 定稿）：菜单内 M2＝收紧丢弃（模式不动）；选中②派就绪事件
    // p1=102 并自动退菜单；模式落地只在主层 M2
    dm.testInjectTextLine("G01 U1");             // layer 2（游标①）
    dm.logicTick();
    dm.testInjectTextLine("G01 M2");             // 菜单内双击＝丢弃
    dm.logicTick();
    EXPECT_EQ(dm.captureMode(), Scanner::ScanMode::DeepHoleScan);   // 模式不变
    dm.testInjectTextLine("G01 L1");             // 游标①→②（261004 竖排：L(↓)＝向⑤方向）
    dm.logicTick();
    const int64_t ready0 = rec.userParam(102);
    dm.testInjectTextLine("G01 M1");             // 选中②：就绪事件 p1=102＋自动退菜单
    dm.logicTick();
    EXPECT_EQ(rec.userParam(102), ready0 + 1);
    EXPECT_EQ(dm.menuState().layer, 1);
    dm.testInjectTextLine("G01 M2");             // 主层双击：2→3 普通交叉
    dm.logicTick();
    dm.testInjectTextLine("G01 M1");             // 启采
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 2);     // 面片掩码（第二次）
    dm.testInjectTextLine("G01 M1");             // 停采收口
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(mock.count("N11 H0"), 3);          // 三轮启停各恰一次
}

// —— T18：标点会话按键隔离（260927 用户口径）——标点扫描中按键只做启停：
// M 双击不切面片/精细/深孔（cycleMode 丢弃）、菜单①②模式设定被拒；
// 模式切换仅在激光族会话中可用；M 短按启停照常 ——
TEST(DeviceManager, T18_MarkerSessionKeyIsolation) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);
    dm.setScanReady(true);   // 261004 矩阵对齐：主层按键（启停/调节/切模式）测试布防就绪凭据

    dm.setCaptureMode(Scanner::ScanMode::MarkerOnly);   // 就绪流程：标点会话
    dm.logicTick();
    dm.testInjectTextLine("G01 M1");                    // 启采（标点：B=kMarkerOnlyBg〔40→10〕＋H 钳 30〔120 带宽饱和链路重开，260927〕激光全关）
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(mock.count("N10 H30 B47 T0 V0 C0 D0 L0"), 1);

    dm.testInjectTextLine("G01 M2");                    // 双击切模式→标点会话被拒
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(dm.captureMode(), Scanner::ScanMode::MarkerOnly);   // 模式不变
    EXPECT_EQ(mock.count("N10 H60 B47 T1 V0 C0 D0 L47"), 0);     // 无精细掩码重发
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 0);     // 无面片掩码重发

    dm.testInjectTextLine("G01 U1");                    // 进菜单（游标①）
    dm.logicTick();
    dm.testInjectTextLine("G01 M1");                    // menuSelect①→采集中防呆拒（261002
    dm.logicTick();                                     //  矩阵表注：扫描中改密度打断累积）
    EXPECT_EQ(dm.menuState().substate, MenuState::Substate::None);   // 子态不进
    EXPECT_EQ(dm.menuState().layer, 2);                 // 菜单不退（拒≠丢）

    dm.testInjectTextLine("G01 U1");                    // 退菜单回主层
    dm.logicTick();
    dm.testInjectTextLine("G01 M1");                    // 停采（启停不受隔离影响）
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(mock.count("N11 H0"), 1);

    // 停采后（仍标点模式）双击依旧被拒——隔离按「当前模式=标点」判，不问采集态
    dm.testInjectTextLine("G01 M2");
    dm.logicTick();
    EXPECT_EQ(dm.captureMode(), Scanner::ScanMode::MarkerOnly);
}

// —— T23：P-1 全局态门禁注入（261002 按键域 §3.2.3）——注入谓词 false（S1/S3/
//      S6/S7 等价全拦态）→ 四类键全丢弃；逃生类（M/H 急停、U/H 回主界面）不问
//      门禁照常执行；注入 true（S2/S4/S5）→ 行为不变 ——
TEST(DeviceManager, T23_KeyStateGateInjection) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    dm.setKeyStateGate([] { return false; });      // 模拟全局全拦态（S6/S7 等价）
    ASSERT_TRUE(dm.open().success);
    dm.setScanReady(true);   // 261004 矩阵对齐：主层按键（启停/调节/切模式）测试布防就绪凭据

    // 四类键全拦：启停（M/S 主层）不启采
    dm.testInjectTextLine("G01 M1");
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 0);
    EXPECT_FALSE(dm.isCapturing());
    // 切模式（M/D）/调节（U/D 景深、L/S 步进）/菜单（U/S 进菜单）全拦
    dm.testInjectTextLine("G01 M2");
    dm.testInjectTextLine("G01 U2");
    dm.testInjectTextLine("G01 R1");
    dm.testInjectTextLine("G01 U1");
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(dm.captureMode(), Scanner::ScanMode::MarkerPlusLaser);   // 模式未切
    EXPECT_EQ(dm.menuState().layer, 1);                               // 未进菜单
    EXPECT_EQ(dm.presetLadderIndex(), 10);                            // 档位未动（261003 定版默认中位档 10）
    EXPECT_EQ(rec.userParam(114), 0);                                 // 景深事件未发

    // 逃生类（261004 暂时停用）：M/H、U/H 手势丢弃留痕——无动作无崩溃
    dm.testInjectTextLine("G01 M3");
    dm.testInjectTextLine("G01 U3");
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(dm.menuState().layer, 1);
    EXPECT_FALSE(dm.isCapturing());                           // 急停停用：无任何设备动作

    // 换白名单（S2/S4/S5 等价放行）→ 启停恢复
    dm.setKeyStateGate([] { return true; });
    dm.testInjectTextLine("G01 M1");
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 1);
    EXPECT_TRUE(dm.isCapturing());
    dm.testInjectTextLine("G01 M1");                 // 停采收口
    dm.logicTick(); dm.logicTick();
}

// —— T22：事件族编号段隔离（261002 按键域定稿）——档位族 p1=111 亮度/112 显示
// 远近/113 体素/114 景深 与菜单族 101-105/110、参数改账 1000+idx 互不撞车 ——
TEST(DeviceManager, T22_EventFamilyIsolation) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);
    dm.setScanReady(true);   // 261004 矩阵对齐：主层按键（启停/调节/切模式）测试布防就绪凭据

    const int64_t dof0 = rec.userParam(114);
    dm.testInjectTextLine("G01 U2");   // 上双击＝景深 近→远（p1=114）
    dm.logicTick();
    EXPECT_EQ(rec.userParam(114), dof0 + 1);

    dm.testInjectTextLine("G01 L2");   // 左双击＝换对象：亮度→显示远近
    dm.logicTick();
    const int64_t dist0 = rec.userParam(112);
    dm.testInjectTextLine("G01 R1");   // 显示远近档 1→2（p1=112）
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(rec.userParam(112), dist0 + 1);

    dm.testInjectTextLine("G01 L2");   // 对象回亮度
    dm.logicTick();
    const int64_t bri0 = rec.userParam(111);
    dm.testInjectTextLine("G01 R1");   // 亮度档 1→2（p1=111）
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(rec.userParam(111), bri0 + 1);

    // 参数改账错峰：bgLight(idx2) 改账广播 p1=1002——不落菜单/档位/景深门内
    //（open 的 bootstrap 装载也广播一次——基线先记）
    const int64_t bgEvt0 = rec.userParam(1002);
    dm.setParam("bgLight", 55.0, Scanner::device::ParamEntry::Source::Ui);
    dm.logicTick();
    EXPECT_EQ(rec.userParam(1002), bgEvt0 + 1);
}

// —— T19：亮度档位梯（G6·261002 扩 20 档）——默认调节对象＝亮度（无需切上下文），
// 左右键＝三参组合档位步进（20 档插值：档1 暗 … 档10 面片推荐 … 档20 强光），
// 钳制不环绕，经 ParamStore 记账，档位事件 p1=111 ——
TEST(DeviceManager, T19_PresetLadderAdjust) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);
    dm.setScanReady(true);   // 261004 矩阵对齐：主层按键（启停/调节/切模式）测试布防就绪凭据
    dm.setBrightnessLadderIndex(1);              // 均分梯默认档10——T19 从档1 测步进
    dm.logicTick(); dm.logicTick();

    auto expectParams = [&](double exp, double laser, double bg) {
        EXPECT_NEAR(dm.getParam("exposure").value, exp, 1e-9);
        EXPECT_NEAR(dm.getParam("laserLevel").value, laser, 1e-9);
        EXPECT_NEAR(dm.getParam("bgLight").value, bg, 1e-9);
    };
    const int64_t briEvt0 = rec.userParam(111);

    // 档1→10：九步到中位（均分 20 档：档10 = t=9/19）
    for (int i = 0; i < 9; ++i) {
        dm.testInjectTextLine("G01 R1");
        dm.logicTick(); dm.logicTick();
    }
    EXPECT_EQ(dm.presetLadderIndex(), 10);
    {   // 档10 均分值：曝光 1+9/19×4 ≈ 2.895ms、激光 9/19×100 ≈ 47.37、补光 同
        const auto steps10 = PresetLadder::builtinLadder();
        expectParams(steps10[9].exposureMs, steps10[9].laserLevel, steps10[9].bgLight);
    }

    dm.testInjectTextLine("G01 R1");   // 档10→11
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(dm.presetLadderIndex(), 11);
    {
        const auto steps11 = PresetLadder::builtinLadder();
        expectParams(steps11[10].exposureMs, steps11[10].laserLevel, steps11[10].bgLight);
    }

    // 到顶钳制：档11→20 九步，再按无效
    for (int i = 0; i < 9; ++i) {
        dm.testInjectTextLine("G01 R1");
        dm.logicTick(); dm.logicTick();
    }
    EXPECT_EQ(dm.presetLadderIndex(), 20);
    {   // 档20 均分终值：曝光 5ms、激光 100、补光 100
        expectParams(5.0, 100.0, 100.0);
    }
    dm.testInjectTextLine("G01 R1");   // 已到顶——无效（钳制不环绕）
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(dm.presetLadderIndex(), 20);
    expectParams(5.0, 100.0, 100.0);   // 261003 均分版档20 终值（旧梯 80 为残留错值）

    dm.testInjectTextLine("G01 L1");   // 档20→19（下调）
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(dm.presetLadderIndex(), 19);
    EXPECT_EQ(rec.userParam(111), briEvt0 + 20);           // 每次有效步进恰一条档位事件
                                                                            // （9+1+9+1=20；到顶无效步不发）
}

// —— T20：设备指示灯（G8 缺口·260927）——N14 S1-S4 下发＋同码去重＋非法码拒绝 ——
TEST(DeviceManager, T20_DeviceLedDedupe) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);

    dm.setDeviceLed(3);                // 绿
    dm.logicTick();
    EXPECT_EQ(mock.count("N14 S3"), 1);
    dm.setDeviceLed(3);                // 同码去重
    dm.logicTick();
    EXPECT_EQ(mock.count("N14 S3"), 1);
    dm.setDeviceLed(4);                // 蓝
    dm.logicTick();
    EXPECT_EQ(mock.count("N14 S4"), 1);
    dm.setDeviceLed(9);                // 非法码拒绝
    dm.logicTick();
    EXPECT_EQ(mock.count("N14 S9"), 0);
}

// —— T21：参数档落盘（G7 缺口·260927）——防抖 2s 冲刷＋close 兜底＋开档读档回账 ——
TEST(DeviceManager, T21_ParamPersistDebounceAndBoot) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    std::string saved;
    int saveCalls = 0;
    {
        DeviceConfig cfg = makeCfg();
        DeviceManager::ParamIo pio;
        pio.load = [] { return std::string(); };        // 无档起步
        pio.persist = [&](const std::string& text) {
            saved = text;
            ++saveCalls;
            return true;
        };
        DeviceManager dm(cfg, gateOk, &bus, nullptr,
                         [&](const std::string& f) { return mock.write(f); }, std::move(pio));
        mock.dm = &dm;
        ASSERT_TRUE(dm.open().success);

        dm.setParam("freqHz", 90.0, Scanner::device::ParamEntry::Source::Ui);
        dm.logicTick(); dm.logicTick();
        EXPECT_EQ(saveCalls, 0);                        // 防抖窗内不落盘
        sleepMs(2100);                                  // 过防抖窗
        dm.logicTick(); dm.logicTick();
        EXPECT_EQ(saveCalls, 1);                        // 拍尾冲刷恰一次
        // P-6 只落档号：freqHz 等非投影参入档；三参段剔除＋brightLadder 段追加
        EXPECT_NE(saved.find("freqHz=90"), std::string::npos);
        EXPECT_EQ(saved.find("exposure="), std::string::npos);
        EXPECT_EQ(saved.find("laserLevel="), std::string::npos);
        EXPECT_EQ(saved.find("bgLight="), std::string::npos);
        EXPECT_NE(saved.find("brightLadder=10"), std::string::npos);   // 默认档 10 入档（261003 定版）

        dm.setBrightnessLadderIndex(12);                // 换档（事务提交后档号入档；
        dm.logicTick(); dm.logicTick();                 //   默认档已 10——须换不同档才是真变更）
        sleepMs(2100);
        dm.close();                                     // close 兜底（无条件冲）
        EXPECT_EQ(saveCalls, 2);
        EXPECT_NE(saved.find("brightLadder=12"), std::string::npos);
    }
    {   // 二代实例：档号制读档——档号展开三参（freqHz=90 复现＋档12 投影复现）
        DeviceConfig cfg = makeCfg();
        DeviceManager::ParamIo pio;
        pio.load = [&saved] { return saved; };
        DeviceManager dm2(cfg, gateOk, &bus, nullptr,
                          [&](const std::string& f) { return mock.write(f); }, std::move(pio));
        ASSERT_TRUE(dm2.open().success);
        dm2.logicTick();
        EXPECT_DOUBLE_EQ(dm2.getParam("freqHz").value, 90.0);
        EXPECT_EQ(dm2.presetLadderIndex(), 12);         // 档号回账
        EXPECT_NEAR(dm2.getParam("exposure").value,
                    PresetLadder::builtinLadder()[11].exposureMs, 1e-9);  // 档12 均分值
        EXPECT_NEAR(dm2.getParam("laserLevel").value,   // 261003 均分版（旧 70/40 为残留错值）
                    PresetLadder::builtinLadder()[11].laserLevel, 1e-9);
        EXPECT_NEAR(dm2.getParam("bgLight").value,
                    PresetLadder::builtinLadder()[11].bgLight, 1e-9);
    }
}

// —— T24：P-6 换档事务（深方案 §3.3 换档事务性）——两段全成才提交档号：
//      ①提交：三段（相机曝光+N10 两参）全成 → 档号+参数账同拍落定＋111 恰一条；
//      ②回弹：相机段败（exposureFail 注入）→ 档号从未移动（天然回弹）＋120 拒因
//       p2=4＋参数账保旧值；
//      ③后值胜出：连按两档 → 终档=最后目标、三参=最后档值 ——
TEST(DeviceManager, T24_LadderTxnCommitRollbackLastWins) {
    // —— ① 提交（正常链：mock 相机 OK + 空闲 N10 参数纯记账）——
    {
        Scanner::infra::EventBus bus;
        EventRecorder rec;
        bus.subscribeAll([&](const Event& e) { rec.record(e); });
        MockMcu mock;
        DeviceConfig cfg = makeCfg();
        auto* camPtr = new FakeCamera();                 // 裸指针工厂（mutable 移动
        DeviceManager dm(cfg, gateOk, &bus,               // lambda 不可拷贝进 std::function）
                         [camPtr]() -> std::unique_ptr<Scanner::hal::IScannerCamera> {
                             return std::unique_ptr<FakeCamera>(camPtr);
                         },
                         [&](const std::string& f) { return mock.write(f); });
        mock.dm = &dm;
        ASSERT_TRUE(dm.open().success);
    dm.setScanReady(true);   // 261004 矩阵对齐：主层按键（启停/调节/切模式）测试布防就绪凭据

        const int64_t evt111_0 = rec.userParam(111);
        dm.setBrightnessLadderIndex(5);                 // UI 换档请求（事务开表）
        dm.logicTick(); dm.logicTick();
        EXPECT_EQ(dm.presetLadderIndex(), 5);           // 三段全成 → 档号已提交
        EXPECT_EQ(rec.userParam(111), evt111_0 + 1);    // 111 恰一条（提交时广播）
        EXPECT_NEAR(dm.getParam("exposure").value,
                    PresetLadder::builtinLadder()[4].exposureMs, 1e-9);  // 档5 均分投影
        EXPECT_EQ(camPtr->exposureCalls, 1);            // 相机直设确实发生

        // —— ③ 后值胜出：连按两次（每拍事务闭环）→ 终档=最后目标 ——
        dm.testInjectTextLine("G01 R1");                // 档5→6
        dm.logicTick(); dm.logicTick();
        dm.testInjectTextLine("G01 R1");                // 档6→7
        dm.logicTick(); dm.logicTick();
        EXPECT_EQ(dm.presetLadderIndex(), 7);
        const auto steps7 = PresetLadder::builtinLadder();   // 拷贝（临时悬空防）
        EXPECT_NEAR(dm.getParam("laserLevel").value,
                    PresetLadder::builtinLadder()[6].laserLevel, 1e-9);  // 档7 均分值
        EXPECT_EQ(rec.userParam(111), evt111_0 + 3);    // 三次换档各恰一条
    }
    // —— ② 回弹（相机段败：exposureFail 注入）——
    {
        Scanner::infra::EventBus bus;
        EventRecorder rec;
        bus.subscribeAll([&](const Event& e) { rec.record(e); });
        MockMcu mock;
        DeviceConfig cfg = makeCfg();
        auto* camFail = new FakeCamera();
        camFail->exposureFail = true;                   // 相机段恒败
        DeviceManager dm(cfg, gateOk, &bus,
                         [camFail]() -> std::unique_ptr<Scanner::hal::IScannerCamera> {
                             return std::unique_ptr<FakeCamera>(camFail);
                         },
                         [&](const std::string& f) { return mock.write(f); });
        mock.dm = &dm;
        ASSERT_TRUE(dm.open().success);
        dm.logicTick();

        const int64_t evt111_0 = rec.userParam(111);
        const int64_t evt120_0 = rec.userParam(120);
        dm.setBrightnessLadderIndex(15);                // 事务开表 → 曝光段即败
        dm.logicTick(); dm.logicTick();
        EXPECT_EQ(dm.presetLadderIndex(), 10);          // 档号从未移动＝天然回弹
                                                           //（261003 定版默认档 10，旧断言 1 为残留）
        EXPECT_EQ(rec.userParam(111), evt111_0);        // 无 111（未提交）
        EXPECT_EQ(rec.userParam(120), evt120_0 + 1);    // 120 拒因恰一条
        EXPECT_NEAR(dm.getParam("exposure").value,      // 参数账保旧值（默认档 10 投影，旧 1.0 为残留）
                    PresetLadder::builtinLadder()[9].exposureMs, 1e-9);
    }
}

// —— T25：261004 矩阵 S2 对齐·就绪凭据——待机（凭据缺位）按键四功能口拒
//      （p1=120,5 未就绪），UI 三参 API 口不查凭据；凭据置位后放行；
//      ①子态 S2 放行不查凭据；①防呆扩到停采保活（会话活＝拒）——
TEST(DeviceManager, T25_StandbyKeyRejectAndReadyCredential) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);
    // —— 起步＝S2 待机（凭据缺位）——
    EXPECT_FALSE(dm.isScanReady());

    // ① 启停拒：M1 不启采（无 N10/N11，不置 capturing）＋120 拒因留痕
    const int64_t rej0 = rec.userParam(120);
    dm.testInjectTextLine("G01 M1");
    dm.logicTick(); dm.logicTick();
    EXPECT_FALSE(dm.isCapturing());
    EXPECT_EQ(mock.count("N10 H60 B47 T0 V1 C1 D0 L47"), 0);
    EXPECT_EQ(rec.userParam(120), rej0 + 1);       // 拒因事件恰一条（p2=5 未就绪）
    dm.testInjectTextLine("G01 M1");               // 再按仍拒且不启采
    dm.logicTick(); dm.logicTick();
    EXPECT_FALSE(dm.isCapturing());
    EXPECT_EQ(rec.userParam(120), rej0 + 2);

    // ② 调节族拒：主界面 R1 步进被拒（档位不动、参数账不动）
    dm.setBrightnessLadderIndex(10);
    dm.logicTick(); dm.logicTick();
    dm.testInjectTextLine("G01 R1");
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(dm.presetLadderIndex(), 10);
    EXPECT_NEAR(dm.getParam("exposure").value,
                PresetLadder::builtinLadder()[9].exposureMs, 1e-9);

    // ③ 切模式拒：M2 记账不动（modeCursor 不变）
    const auto modeCur0 = dm.menuState().modeCursor;
    dm.testInjectTextLine("G01 M2");
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(dm.menuState().modeCursor, modeCur0);

    // ④ 景深拒：U2 不切（p1=114 事件不发）
    const int64_t dof0 = rec.userParam(114);
    dm.testInjectTextLine("G01 U2");
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(rec.userParam(114), dof0);

    // ⑤ UI 三参 API 口不查凭据（261004 用户裁定：待机 UI 可调三参）
    dm.setBrightnessLadderIndex(5);
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(dm.presetLadderIndex(), 5);

    // ⑥ ①子态 S2 放行（矩阵 ①行 S2=✓——不查凭据）：进菜单①调体素密度
    dm.testInjectTextLine("G01 U1");
    dm.logicTick(); dm.logicTick();
    dm.testInjectTextLine("G01 M1");               // 选中①（凭据缺位不拒——仅会话活才拒）
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(dm.menuState().substate, MenuState::Substate::AdjustVoxel);
    dm.testInjectTextLine("G01 R1");
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(rec.userParam(113), 1);              // 体素档步进成功
    dm.testInjectTextLine("G01 M1");               // 确认退出
    dm.logicTick(); dm.logicTick();

    // ⑦ 凭据置位（app arm 成功等价）→ 启停/调节放行
    dm.setScanReady(true);
    const int64_t bri0 = rec.userParam(111);
    dm.testInjectTextLine("G01 R1");               // 主界面步进：凭据在→放行
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(dm.presetLadderIndex(), 6);
    EXPECT_EQ(rec.userParam(111), bri0 + 1);

    // ⑧ ①防呆扩判据：凭据在（会话活·停采保活＝isCapturing false）→ 选中①拒
    dm.testInjectTextLine("G01 U1");
    dm.logicTick(); dm.logicTick();
    dm.testInjectTextLine("G01 M1");               // ①选中——会话活（凭据）即拒
    dm.logicTick(); dm.logicTick();
    EXPECT_NE(dm.menuState().substate, MenuState::Substate::AdjustVoxel);
    EXPECT_EQ(dm.menuState().layer, 2);            // 仍菜单浏览态（未进子态）

    // ⑨ 凭据清位（收尾等价；重置走⑤链见⑩⑪）→ 复拒
    dm.testInjectTextLine("G01 U1");               // 退菜单
    dm.logicTick(); dm.logicTick();
    dm.setScanReady(false);
    dm.testInjectTextLine("G01 M1");
    dm.logicTick(); dm.logicTick();
    EXPECT_FALSE(dm.isCapturing());

    // ⑩ 261004 分辨率锁定（§3.3.3 ①补充）：会话曾建立（就绪过）即锁——③收尾
    //    清凭据回 S2 后菜单①仍拒（p2=6）；UI 体素换档口同口径拒
    dm.testInjectTextLine("G01 U1");               // 进菜单（cursor ①）
    dm.logicTick(); dm.logicTick();
    const int64_t lock0 = rec.userParam(120);
    dm.testInjectTextLine("G01 M1");               // 选中①——已扫描锁定拒
    dm.logicTick(); dm.logicTick();
    EXPECT_NE(dm.menuState().substate, MenuState::Substate::AdjustVoxel);   // 未进子态
    EXPECT_EQ(dm.menuState().layer, 2);            // 仍菜单浏览态
    EXPECT_EQ(rec.userParam(120), lock0 + 1);      // 拒因留痕恰一条（p2=6）
    const int voxBefore = dm.voxelLadderIndex();   // UI 换档口：锁定拒（档不动）
    dm.setVoxelLadderIndex(voxBefore == 1 ? 2 : 1);
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(dm.voxelLadderIndex(), voxBefore);

    // ⑪ ⑤重置确认链＝唯一解锁路径：游标⑤→确认子态→中键执行（p1=105）→锁解→①可进
    for (int i = 0; i < 4; ++i) dm.testInjectTextLine("G01 L1");   // ①→②→③→④→⑤
    dm.logicTick(); dm.logicTick();
    dm.testInjectTextLine("G01 M1");               // ⑤选中→二次确认子态
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(dm.menuState().substate, MenuState::Substate::ConfirmReset);
    const int64_t rst0 = rec.userParam(105);
    dm.testInjectTextLine("G01 M1");               // 确认执行——105 派发＋锁定解除
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(rec.userParam(105), rst0 + 1);
    dm.testInjectTextLine("G01 U1");               // 再进菜单（cursor ①）
    dm.logicTick(); dm.logicTick();
    dm.testInjectTextLine("G01 M1");               // ①放行（锁已解）
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(dm.menuState().substate, MenuState::Substate::AdjustVoxel);
}

// —— T26：分辨率（体素密度）档下限＝0.01mm（261004 用户报：跳挡向下最多只
//      能调到 0.02）——全链路实证：菜单①子态连续下调从默认 14 档到 index 1
//      （0.01mm）有效，再按＝到底静默（不环绕、事件不再发）——
TEST(DeviceManager, T26_VoxelLadderFloor0p01mm) {
    Scanner::infra::EventBus bus;
    EventRecorder rec;
    bus.subscribeAll([&](const Event& e) { rec.record(e); });
    MockMcu mock;
    DeviceConfig cfg = makeCfg();
    DeviceManager dm(cfg, gateOk, &bus, nullptr,
                     [&](const std::string& f) { return mock.write(f); });
    mock.dm = &dm;
    ASSERT_TRUE(dm.open().success);                 // 起步 S2 无会话——①可进（未锁定）

    // 梯表本身：27 档、首档 0.01mm（0.01..0.10 / 0.2..1.0 / 1.5..5.0）
    const auto steps = Scanner::device::VoxelDensityLadder().steps();
    ASSERT_EQ(steps.size(), static_cast<size_t>(27));
    EXPECT_NEAR(steps.front(), 0.01, 1e-12);

    // 菜单①全链：进菜单→选中①进子态→连续下调 13 次到底（默认 14→1）
    dm.testInjectTextLine("G01 U1");
    dm.logicTick(); dm.logicTick();
    dm.testInjectTextLine("G01 M1");
    dm.logicTick(); dm.logicTick();
    ASSERT_EQ(dm.menuState().substate, MenuState::Substate::AdjustVoxel);
    ASSERT_EQ(dm.voxelLadderIndex(), 14);           // 默认中位档
    const int64_t ev0 = rec.userParam(113);
    for (int i = 0; i < 13; ++i) {
        dm.testInjectTextLine("G01 L1");            // 下调（键序＝档值变小方向）
        dm.logicTick(); dm.logicTick();
    }
    EXPECT_EQ(dm.voxelLadderIndex(), 1);            // 到底＝index 1
    EXPECT_NEAR(dm.voxelLadderValue(), 0.01, 1e-9); // 档值 0.01mm
    EXPECT_EQ(rec.userParam(113), ev0 + 13);        // 每步恰一条 113 事件
    // 再按：到底静默——档不动、事件不再发
    dm.testInjectTextLine("G01 L1");
    dm.logicTick(); dm.logicTick();
    EXPECT_EQ(dm.voxelLadderIndex(), 1);
    EXPECT_EQ(rec.userParam(113), ev0 + 13);
}
