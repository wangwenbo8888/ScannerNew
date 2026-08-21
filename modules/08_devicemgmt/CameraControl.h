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
    double defaultExposureMs = 10.0;
    std::string triggerSource = "Line2";
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
    std::atomic<double> m_currentExposureMs{10.0};
    std::atomic<double> m_currentGain{0.0};      // 按 GainRaw 原生单位传，dB 语义由上层换算

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
    void stopSideCapture(int sideIndex);
    void applySideParams(int sideIndex);
};

} // namespace Scanner::device
