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
    };

    // 侧缓冲
    struct SideBuffer {
        cv::Mat image;
        uint64_t frameId = 0;
        std::atomic<bool> ready{false};
    };

    StereoPairConfig m_config;
    std::atomic<bool> m_isOpen{false};
    std::atomic<bool> m_isCapturing{false};
    std::atomic<double> m_currentExposureMs{25.0};  // 与 DeviceManager specs 默认同源
    std::atomic<double> m_currentGain{0.0};      // 按 GainRaw 原生单位传，dB 语义由上层换算

    // 帧号稳定偏移配对（260911）：N10 重发/启停后某侧多吃漏吃一触发沿→L-R 恒差
    // ±1 永不收敛——连续 30 次同号失配即采纳，按时间对齐配对（严格等值期间=0）
    int64_t m_pairOffset = 0;                    // 仅在 m_bufferMutex 内触碰
    int64_t m_lastMismatchOff = 0;
    int m_mismatchStreak = 0;

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
