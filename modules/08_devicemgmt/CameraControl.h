#pragma once
// ============================================================================
// CameraControl.h — 大恒 Galaxy SDK 相机采集封装
//
// 实现 Scanner::hal::IScannerCamera 接口，管理一对立体相机（左 + 右）。
// 线程安全：SDK 回调线程 → 内部 mutex → std::function 回调。
// ============================================================================

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "IScannerCamera.h"
#include "GalaxyIncludes.h"

#include <mutex>
#include <atomic>

namespace Scanner::device {

// ============================================================================
// 相机对配置
// ============================================================================
struct StereoPairConfig {
    int deviceIndexLeft  = 0;
    int deviceIndexRight = 1;
    bool rotateRight180  = true;
    std::string triggerSource = "Line2";   // 硬件触发源（装机接线口径——config/camera.json）
    bool pairStrictFrameId = true;         // 帧号严格配对开关（false=按时间对齐交付——
                                           // 带宽/帧率实测实验用；T/V 奇偶归属不保证）
    bool timestampPairing = true;          // 260927 方案B实验：帧号不等时按时间戳判组
                                           //（|L.ts−R.ts−Δ|<ε=周期/4，Δ 在线 EMA）；
                                           // 时钟若随流重开复位则恒不命中——自然回落
                                           // 丢组/偏移采纳现行路径（无需开关旁路）
};

// ============================================================================
// CameraControl — 单对立体相机控制
// ============================================================================
class CameraControl : public hal::IScannerCamera {
    friend class CaptureEventHandler;

public:
    explicit CameraControl(const StereoPairConfig& config = {});
    ~CameraControl() override;

    CameraControl(const CameraControl&) = delete;
    CameraControl& operator=(const CameraControl&) = delete;

    // IScannerCamera 接口
    std::string getDeviceName() const override;
    std::string getSerialNumber() const override;
    std::string getPlatform() const override { return "Windows"; }

    Result open() override;
    Result close() override;
    bool isOpen() const override;

    Result setExposure(double ms) override;
    Result setGain(double dB) override;
    Result setResolution(int width, int height) override;
    Result setContrast(int leftValue, int rightValue) override;  // 软件端对比度·左右分置
                                                 //（0=直通；±逐帧 ImageImprovment）

    Result setCalibration(const hal::CameraIntrinsics& left,
                          const hal::CameraIntrinsics& right,
                          const hal::StereoExtrinsics& stereo) override;
    hal::CameraIntrinsics getLeftIntrinsics() const override;
    hal::CameraIntrinsics getRightIntrinsics() const override;
    hal::StereoExtrinsics getStereoExtrinsics() const override;

    Result startCapture() override;
    Result stopCapture() override;
    bool isCapturing() const override;

    Result grabFrame(hal::StereoFrame& frame, int timeoutMs = 1000) override;

    Result startAsyncCapture(hal::FrameCallback cb) override;
    Result stopAsyncCapture() override;

    double getTemperature() const override;

    uint64_t frameRollbackCount() const override;   // 双侧帧号回退累计（自动恢复检测源）

    static int enumerateDevices();

private:
    // 单侧资源
    struct SideResource {
        CGXDevicePointer          device;
        CGXStreamPointer          stream;
        CGXFeatureControlPointer   featureControl;
        ICaptureEventHandler*     eventHandler = nullptr;
        bool                      isOpen = false;
        bool                      isCapturing = false;
        cv::Mat                   contrastLut;     // 对比度 LUT（256×1·CV_8UC1——懒建/值变
                                                  // 重建；仅本侧回调线程触碰）
        int                       lutContrast = 0; // LUT 已烘焙值（变更检测）
    };

    // 侧缓冲
    struct SideBuffer {
        cv::Mat image;
        uint64_t frameId = 0;
        uint64_t timestamp = 0;   // 设备时间戳原始单位（GetTimeStamp——方案B配对判据）
        std::atomic<bool> ready{false};
    };

    StereoPairConfig m_config;
    std::atomic<bool> m_isOpen{false};
    std::atomic<bool> m_isCapturing{false};
    std::atomic<double> m_currentExposureMs{3.0};  // 相机侧初始值；账本 spec=exposure{def 10, 1..100}（DeviceManager.cpp makeParamSpecs）
    std::atomic<double> m_currentGain{0.0};      // 按 GainRaw 原生单位传，dB 语义由上层换算
    std::atomic<int> m_contrastL{0};             // 软件对比度·左（0=直通零开销；± [-100,100]）
    std::atomic<int> m_contrastR{0};             // 软件对比度·右（左右分置——260927→1002）

    // 帧号稳定偏移配对（260911）：N10 重发/启停后某侧多吃漏吃一触发沿→L-R 恒差
    // ±1 永不收敛——连续 30 次同号失配即采纳，按时间对齐配对（严格等值期间=0）
    int64_t m_pairOffset = 0;                    // 仅在 m_bufferMutex 内触碰
    int64_t m_lastMismatchOff = 0;
    int m_mismatchStreak = 0;
    std::atomic<bool> m_firstPairPending{false}; // 开流后首组日志武装（260927 方案A
                                                 // 实验——自动恢复成活判据可见性）

    // —— 时间戳配对（260927 方案B实验）——Δ=L.ts−R.ts 两机钟差：两相机各自上电
    // 起算不可直比，但同触发曝光对的钟差近似恒定（晶拖 ppm 级由 EMA 自跟踪）；
    // 链路重开帧号归零而时钟自由跑→Δ 仍有效＝零中断配对（核心假设，真机验证）——
    int64_t m_pairTsDelta = 0;                   // 仅在 m_bufferMutex 内触碰（EMA α=1/16）
    bool m_pairTsValid = false;                  // Δ 已锚定（首个严格等值对起；只由
                                                 // fid 等值对喂——防救回对锁死错误 Δ）
    std::atomic<uint64_t> m_sideTsPeriod[2]{};   // per-side 相邻帧 ts 差（≈触发周期，
                                                 // ε=period/4 单位无关；handler 各侧单写）

    // —— T/V 激光组判定（260927 时间戳奇偶法）：family(t)=parity(round((t−t₀)/
    //    周期))——时间轴数的是 MCU 实际发出的脉冲序，免疫链路重开（BlockID 归零）
    //    与相机漏帧（无帧=轴上无缝隙）。锚 t₀=开流首交付对 ts（假设：N11 H1 后
    //    首脉冲恒打 T 左斜——固件确认中；确认前 tvKnown 恒 false 可整体旁路）——
    uint64_t m_tvT0 = 0;                         // 锚（m_bufferMutex 内；首对锚定）
    bool m_tvT0Valid = false;                    // t₀ 已锚定（同上）

    // 标定缓存（注入式 B3：app 从 06 标定结果仓库喂入，08 不做第二真相源）
    mutable std::mutex m_calibMutex;
    hal::CameraIntrinsics m_calibLeft;
    hal::CameraIntrinsics m_calibRight;
    hal::StereoExtrinsics m_calibStereo;

    SideResource m_sides[2];
    SideBuffer   m_buffers[2];

    hal::FrameCallback m_frameCallback;
    mutable std::mutex m_callbackMutex;
    mutable std::mutex m_bufferMutex;  // 保护 tryDeliver 的缓冲区访问

    void startSideCapture(int sideIndex);
    bool stopSideCapture(int sideIndex);   // 返 true=干净停止（AcquisitionStop 成）
    void applySideParams(int sideIndex);
};

} // namespace Scanner::device
