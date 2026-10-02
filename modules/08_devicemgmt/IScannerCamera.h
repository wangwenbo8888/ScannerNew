#pragma once
// ============================================================================
// IScannerCamera.h — 双目相机接口（HAL 层）
//
// 扫描仪相机的抽象接口。实现由平台特定驱动提供（Win/Jetson）。
// ============================================================================

#include "base/types.h"
#include <opencv2/core.hpp>
#include <functional>

namespace Scanner::hal {

// ============================================================================
// 帧数据（双目）
// ============================================================================
struct StereoFrame {
    FrameId frameId = 0;
    FrameId frameIdLeft = 0;    // 左相机原始帧号（调试——相机预览显示用）
    FrameId frameIdRight = 0;   // 右相机原始帧号（同上）
    TimestampMs timestamp = 0;
    FrameId timestampLeft = 0;   // 左相机设备时间戳（原始单位——260927 方案B实验：
                                 // 时间戳配对判据/调试用，与右机时钟不可直比）
    FrameId timestampRight = 0;  // 右相机设备时间戳（同上）
    bool tvKnown = false;        // T/V 判定已知（时间戳奇偶法——CameraControl 计算；
                                 // false=未知〔sim 帧/Δ 未锚定〕消费方自行回退帧号奇偶）
    bool tvLeftSkew = true;      // true=左斜 T 组 / false=右斜 V 组（tvKnown 时有效；
                                 // 锚：开扫首帧=T〔N11 H1 后首脉冲恒 T，固件确认中〕）
    cv::Mat leftGray;   // 左图灰度 CV_8UC1
    cv::Mat rightGray;  // 右图灰度 CV_8UC1
};

// ============================================================================
// 相机参数
// ============================================================================
struct CameraIntrinsics {
    cv::Mat cameraMatrix;   // 3x3 内参
    cv::Mat distCoeffs;     // 畸变系数
    cv::Size imageSize;
};

struct StereoExtrinsics {
    cv::Mat R, T;           // 右相对左
    cv::Mat R1, R2, P1, P2, Q;  // 立体校正
};

// ============================================================================
// 帧回调
// ============================================================================
using FrameCallback = std::function<void(const StereoFrame&)>;

// ============================================================================
// IScannerCamera — 双目相机接口
// ============================================================================
class IScannerCamera {
public:
    virtual ~IScannerCamera() = default;

    // 设备信息
    virtual std::string getDeviceName() const = 0;
    virtual std::string getSerialNumber() const = 0;

    // 连接/断开
    virtual Result open() = 0;
    virtual Result close() = 0;
    virtual bool isOpen() const = 0;

    // 参数
    // 注：帧率=N10 H 拍照频率（下位机管）；激光归 MCU——2026-08-20 08 设计 §5.1 裁决
    virtual Result setExposure(double ms) = 0;
    virtual Result setGain(double dB) = 0;
    virtual Result setResolution(int width, int height) = 0;
    // 对比度（260927 增→1002 左右分置）：软件端图像增强（Galaxy ImageImprovment/
    // SetContrastParam——设备端无此特性）。0=直通零开销（默认）；>0 增强/<0 减弱，
    // 建议域 [-100,100]；左右各自独立值，逐帧生效。启动默认值自 config/camera.json
    // "contrastLeft"/"contrastRight" 装载
    virtual Result setContrast(int leftValue, int rightValue) = 0;

    // 标定参数（注入式，08 设计 B3 修正）：app 从 06 标定结果仓库取已解析内外参喂入——
    // 08 不带 json 解析器、不做第二真相源（标定数据归属 06）
    virtual Result setCalibration(const CameraIntrinsics& left,
                                  const CameraIntrinsics& right,
                                  const StereoExtrinsics& stereo) = 0;
    virtual CameraIntrinsics getLeftIntrinsics() const = 0;
    virtual CameraIntrinsics getRightIntrinsics() const = 0;
    virtual StereoExtrinsics getStereoExtrinsics() const = 0;

    // 采集控制
    virtual Result startCapture() = 0;
    virtual Result stopCapture() = 0;
    virtual bool isCapturing() const = 0;

    // 同步采集（阻塞/超时）
    virtual Result grabFrame(StereoFrame& frame, int timeoutMs = 1000) = 0;

    // 异步采集（回调）
    virtual Result startAsyncCapture(FrameCallback cb) = 0;
    virtual Result stopAsyncCapture() = 0;

    // 温度
    virtual double getTemperature() const = 0;

    // 帧号回退累计（同侧 BlockID 下降＝链路自发重开出流指纹；缺省 0=不支持）——
    // DeviceManager 链路自动恢复（260927 方案A实验）检测源
    virtual uint64_t frameRollbackCount() const { return 0; }

    // 平台
    virtual std::string getPlatform() const = 0;  // "Windows" / "Jetson"
};

} // namespace Scanner::hal
