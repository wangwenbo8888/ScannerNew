#pragma once
// ============================================================================
// DeviceManager.h — 设备管理门面 + 逻辑线程（协议 260831 批1-C 口径）
//
// 组合根：MCUDriver/KeySemantics/MenuLogic/ParamStore/WarmupSequence/
// ModeController 全 unique_ptr 内部持有（铁规：不漏零件指针/类型——对外只出值
// 类型与薄转发；MCUDriver/KeySemantics 仅前向声明）。门禁回调注入不反链 07/10
//（GateQuery 只进不出）。
//
// 线程纪律（设计 §6.1「切模式经队列转逻辑线程执行」）：
//   - 无锁零件（CommandChannel/ParamStore map/MenuLogic/Warmup）单一属主=逻辑
//     线程；门面一切变异入口经 post() 任务队列编队（mutex+deque，容量 64 满丢新
//     +warn），logicTick 开头排空执行；manualTick 模式下测试调 logicTick 即驱动。
//   - 同步返回值口径：门禁检查（ModeController::request）在调用方线程同步做，
//     拒→不入队同步返回 fail；过→命令编队执行，返回 request 结果（组成败异步经
//     Fault/StateChanged 观测）。相机三口的返回值=前置检查（无相机同步 fail，
//     实际动作编队异步）。
//   - open() 特例：N12 温度周期在起逻辑线程**前**发（盲发无 ACK）；close() 特例：
//     停线程后余任务丢弃（退出场景不保送）。
//   - 参数双口：getParam 读互斥保护快照（logicTick 每拍全量拷）；setParam 编队调
//     账本 setValue。menuState()/getLastTemperatures() 同款互斥快照。
//
// 逻辑线程一拍（logicTick，!manualTick 时 10ms 循环）：
//   drain 任务队列 → pump 上行 3 环 → 手势派发（G01 已判 S/D/H → KeySemantics
//   → 状态转移表）→ Warmup tick → 故障巡检（相机掉线边沿/串口无声/G01 手势环
//   溢——温度双警在 onTemp 回调内）→ 参数/菜单/温度快照 → 自检状态机推进。
//
// 协议 260831 口径（批1-C）：
//   - 下行盲发（无 ACK 无回显，D8）：N10 采集参数七参（即启采）/ N11 H0 停止
//     熄灯 / N12 T 温度上报周期 / N13 S 加热目标（0-80 钳）。命令完成回调同步
//     回告写成败；组步链退化为单步链（框架保留给采集用）。
//   - 上行 G01/G02/G03 三环（MCUDriver 分流）：G01 手势（MCU 已判，PC 无判定
//     ——KeyManager 已退役）→ pendingGestures_ 暂存 → logicTick 派发；G02 四路
//     温度（双警+Warmup 喂入+记账）；G03 触发计数（记账在 MCUDriver lastShot_，
//     门面不转发）。
//   - 启采=N10 本身（参数+灯控+启动）；停采=N11 H0；切模式（标定/待机）无专属
//     下行命令——纯软件落板。
//   - 自检（startupSelfCheck）：mcuLink 凭据=lastRxTime>0（N12 周期 G02 到达
//     即活）；bgLight/laser 两段式闪灯（N10 H5 短采→保持 800ms→N11 H0；盲发
//     无回显，发毕即报）；camera 项以相机 open 状态判定。
//
// 口径裁定（沿承）：
//   - ctor 第 5 参 serialWriteOverride：MCUDriver 测试缝透传（std::function 直传，
//     不泄 MCUDriver 类型；产线空=真串口）；
//   - testInjectRaw：门面级测试缝（等价 rx 线程收到原始字节）；
//   - menuSelect ③/④ 出口：EventBus UserDefined（param1=cursor）——待 base 增
//     专用事件类型（人工过会后改）；①/② 仅日志（modeCursor 账本为准）；
//   - 模式落板广播：EventBus StateChanged（param1=新态）；same-mode 不广播；
//   - ParamStore onParamChanged → EventBus UserDefined（param1=参数索引=specs
//     登记序号，sourceId=8 标 08 来源）；
//   - 按键门禁（KeySemantics gate）= !isCapturing（M1：采集态菜单/模式/调节键
//     丢弃；启停键不问门禁）；
//   - N10 组参 Dispatch：采集中全参重发；空闲仅记账 done(true,false)；exposure
//     相机直设，无相机=纯记账 done(true,true)。
// ============================================================================

#include "IScannerCamera.h"
#include "MenuLogic.h"        // MenuState（menuState 返回值）
#include "ModeController.h"   // DeviceMode（mode 返回值）
#include "ParamStore.h"       // ParamEntry（getParam 返回值）
#include "WarmupSequence.h"   // WarmupConfig（DeviceConfig 值成员）
#include "base/EventBus.h"
#include "base/types.h"
#include "serial/McuFrame.h"  // TempFrame/GestureEvent（快照与手势派发）

#include <atomic>
#include <chrono>
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

// —— Fault 码表（§6.2；8 码 + 0x081x 开机段，枚举值沿承）——
// 边沿纪律：每类记「上次触发锚」，恢复（心跳到帧/温度回落/相机重开）清锚允许
// 再触发——防爆屏；FaultOccurred 事件 param1=码（sourceId=8）。§6.2「只报不动手」。
enum class DevFault : int64_t {
    CameraLost      = 0x0801,  // #1 相机掉线：任意时刻 isOpen 翻 false 边沿（原开过
                               //     才算；不限采集中；只报不停手）
    SerialSilent    = 0x0802,  // #2≡#10 串口无声=通讯心跳丢失（同源合并）：收到过帧
                               //     （lastRx>0）后停更超 heartbeatTimeoutMs；帧到清锚
    TempOverMax     = 0x0803,  // #3 温度爆表：任一路 >tempMaxC；全路回落清锚
    TempSpike       = 0x0804,  // #4 温度乱跳：相邻 G02 帧同路 |Δ|/Δt>tempSpikeC ℃/s；
                               //     次帧平稳清锚
    WarmupTimeout   = 0x0805,  // #5 预热超时：WarmupSequence onTimeout（只报不停加热）
    KeyRingOverflow = 0x0806,  // #6 手势队列挤爆：G01 手势环满丢新计数增长（事件型）
    CmdNoAck        = 0x0807,  // #7 串口写失败（无 ACK 机制——盲发口径 D8）：命令发送
                               //     失败收口归此码，detail 串区分命令名
    SeqGap          = 0x0808,  // #9 闲置——G03 帧对账预留（枚举值保留，批4 接通）
    // —— 表外既有路径（0x081x 开机段）——
    CameraOpenFail  = 0x0810,  // open 一条龙相机打开失败（同步倒序关）
    McuOpenFail     = 0x0811,  // open 一条龙 MCU 串口打开失败（同步倒序关）
};

struct DeviceConfig {
    std::string serialPort;
    int baud = 115200;                          // 协议现状固定波特率
    int tempReportPeriodMs = 100;               // N12 T 温度上报周期（MCUDriver 钳 5-1000）
    WarmupConfig warmup{};
    bool manualTick = false;      // 测试：不起逻辑线程，logicTick() 手动驱动
    // —— 故障巡检阈值（§6.2；产线默认值，测试可注入小值换快用例）——
    int heartbeatTimeoutMs = 10000;  // 串口无声（#2/#10）：距末帧超此值报 0x0802
    double tempMaxC = 60.0;          // 温度爆表（#3）：任一路超此值报 0x0803
    double tempSpikeC = 2.0;         // 温度乱跳（#4）：同路相邻帧速率超此 ℃/s 报 0x0804
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

    // —— 开闭（open 一条龙：相机→MCU→上行接线→参数装载+快照→N12 温度周期→
    //    逻辑线程→DeviceConnected；任一同步步败：倒序关已开+Fault+fail）——
    Result open();
    Result close();

    // —— 切模式三口令（门禁调用方线程同步判；过→编队逻辑线程执行，成功回调才
    //    擦板；260831 标定/待机无专属下行命令——纯软件落板；返回 request 结果）——
    Result enterCalibration();
    Result enterScan();
    Result toIdle();

    // —— 预热（N13 S<目标> + WarmupSequence 看火；稳/超时 done 恰一次、逻辑
    //    线程触发；超时不停加热——停止机制待协议方）——
    void startWarmup(int targetC, std::function<void(bool stable)> done);

    // 启动自检（软件开起跑一次，open 成功后由 app 调）：mcuLink=lastRxTime 凭据
    // + bgLight/laser 两段式闪灯 + 相机 open 验证。全程 post 编队跑逻辑线程；
    // report 每项回投（逻辑线程回调——app 侧自行转发 UI 线程）。
    // key："mcuLink" / "bgLight" / "laser" / "camera"
    void startupSelfCheck(std::function<void(const std::string&, bool)> report);

    // 串口收发监听透传（MCUDriver wireTap；调试弹窗用——open 前设置以捕获探测帧）
    void setWireTap(std::function<void(bool tx, const std::string& data)> tap);

    // —— 观测 ——
    bool isDeviceReady() const;                 // 相机开+MCU 开（无相机工厂=只看 MCU）
    serial::TempFrame getLastTemperatures() const;   // G02 四路（互斥快照）
    bool isCapturing() const;                   // 采集子态真相源（ModeController 黑板）
    DeviceMode mode() const;                    // 模式黑板快照（UI/测试）
    MenuState menuState() const;                // 菜单账本快照（UI 常显）

    // —— 相机薄转发（统一编队：返回值=前置检查，实际动作逻辑线程异步执行）——
    bool isCameraOpen() const;
    Result setCameraExposure(double ms);
    Result startFrameStream(hal::FrameCallback cb);
    Result stopFrameStream();

    // —— 采集启停（不切模式；幂等：黑板同值直返；编队执行——调用后需一拍
    //    logicTick 落地下行帧。260831：启采=N10 本身（七参自账本+灯型组装，
    //    相机先启再发 N10——帧号错位修正口径保持）；停采=N11 H0。
    //    laserOn=false（标点扫描 A 模式）：L=0 且四激光管全 0（只开补光）；
    //    缺省 true=账本全参（面片扫描 B 模式）——
    void startCapture(bool laserOn = true);
    void stopCapture();
    /// 实测相机帧率（配对交付差分 1s 窗口——帧回调链内计数，UI 只读）
    int measuredCameraFps() const { return m_measuredFps.load(std::memory_order_relaxed); }

    // —— 灯光直控（用户按钮直调；编队逻辑线程执行，不启停采集）——
    /// N10 灯字段即时生效（盲发）：bgOn/laserOn=true 取账本值、false 置 0；
    /// 激光管二值映射暂版——laserOn=true → T1V1C0D0，false → 四管全 0（批3 换 ScanMode 映射）
    void setLights(bool bgOn, bool laserOn);

    // —— 打光场景封装（灯型三态；N10 即时生效，组合语义见各自注释）——
    /// 只打补光灯（标志点扫描 A 模式）：B=账本值，L=0，激光管全关
    void lightsBgOnly();
    /// 全灭（补光/激光全关；等价 close 前收口语义）
    void lightsAllOff();

    // —— 参数双口（Critical #1：账本归逻辑线程，引用出口已删）——
    ParamEntry getParam(const std::string& key) const;   // 互斥快照读（轻拷）
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

    // —— 启动自检状态机（逻辑线程私有；logicTick 每 10ms 驱动，无阻塞等待）——
    struct SelfCheckSm {
        int stage = -1;              // -1 闲 / 0 发 N10 / 1 保持 800ms / 2 发 N11 H0 / 3 相机验证
        int item = 0;                // 闪灯两段式：0=bgLight 项（B=账本 L=0）/ 1=laser 项（B=0 L=账本 T1V1C0D0）
        int64_t stageStartMs = 0;
        std::function<void(const std::string&, bool)> report;
        std::atomic<int>  frames{0};             // 相机验证帧计数（相机回调线程写）
        std::atomic<bool> frameValid{false};     // 双目图非空凭据
    } selfCheck_;
    void selfCheckTick(int64_t nowMs_);       // logicTick 末驱动（单次 µs 级；名避让 nowMs()）
    std::mutex openMtx_;                         // open/close 串行化（启动后台线程与
                                                 // ScannerWindow 设备线程可能并发 open）
    void drainPosts();                          // logicTick 开头排空（逻辑线程属主）
    void checkTempFaults(const serial::TempFrame& t);  // 温度双警 0x0803/0x0804（onTemp 内）
    void publishFault(int64_t code, const std::string& detail);
    void publishEvent(EventType t, int64_t p1, int64_t p2);
    void dispatchGesture(const serial::GestureEvent& ev);  // G01 手势 → KeySemantics
    void buildKeyActions();                     // KeySemActions 11 出口的接线
    void applyAdjust(int dir);                  // 调节步进 → MenuLogic+ParamStore
    void sendSeq(std::vector<SeqStep> steps, std::function<void(bool)> onDone);
    void onParamDispatch(const std::string& key, double v, ParamStore::Done done);
    void startStreamIfReady();
    void refreshParamSnapshot();                // 全参数拷入互斥快照
    // 切模式/启停的命令组主体（逻辑线程执行——门禁已在调用方线程过）
    std::vector<SeqStep> captureSeqSteps();      // 采集组步链（单步 N10——启采=N10 本身）
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

    // —— 菜单/温度快照（同款互斥模式；logicTick 末与参数快照同段统一刷新）——
    mutable std::mutex menuSnapMtx_;
    MenuState menuSnap_{};
    mutable std::mutex tempSnapMtx_;
    serial::TempFrame tempSnap_{};

    // —— 运行时记账（逻辑线程属主；跨线程读口走上方快照）——
    hal::FrameCallback frameCb_;                // startFrameStream 登记的帧出口
    std::atomic<uint64_t> m_rxCnt_{0};          // 帧回调计数（帧率差分——相机线程写）
    std::atomic<int> m_measuredFps{0};          // 实测帧率（1s 窗口差分——UI 读）
    std::chrono::steady_clock::time_point m_lastFpsTick_{};  // 差分基点（相机线程）
    serial::TempFrame lastTemps_{};             // 逻辑线程账本（快照源）
    std::function<void(bool)> warmupDone_;      // 当前预热完成回调（onStable/onTimeout 消费）
    bool captureLaserOn_ = true;                // 采集灯型（逻辑线程属主）：false=N10 强制 L=0+四管全 0
    std::vector<serial::GestureEvent> pendingGestures_;   // G01 手势暂存（pump 回调入队，
                                                           // logicTick ③ 派发后清空）
    // —— 故障边沿锚（逻辑线程属主；恢复清锚防复报）——
    bool camFaultLatched_ = false;              // 0x0801 掉线锁（相机重开清锚）
    bool camWasOpen_ = false;                   // 0x0801 前置锚：相机曾开（open 成功即置）
    bool serialSilentLatched_ = false;          // 0x0802 锁（再收到帧清锚）
    bool tempHotLatched_ = false;               // 0x0803 锁（全路回落清锚）
    bool tempSpikeLatched_ = false;             // 0x0804 锁（次帧平稳清锚）
    serial::TempFrame prevTemps_{};             // 0x0804 上一 G02 帧（速率分子/分母）
    bool prevTempsValid_ = false;
    uint64_t lastKeyDrop_ = 0;                  // 0x0806 上拍手势环丢新计数（单调累计对齐）
    bool opened_ = false;
    std::atomic<bool> running_{false};
    std::thread logicThread_;
};

} // namespace Scanner::device
