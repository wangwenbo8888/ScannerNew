// ============================================================================
// CameraControl.cpp — 大恒 Galaxy SDK 相机采集实现
// ============================================================================

#include "CameraControl.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <cstring>

namespace Scanner::device {

// ============================================================================
// SDK 回调处理器 — 在 SDK 内部线程中执行
// ============================================================================
class CaptureEventHandler : public ICaptureEventHandler {
public:
    CaptureEventHandler(CameraControl* owner, int sideIndex, bool rotateRight)
        : m_owner(owner), m_sideIndex(sideIndex), m_rotateRight(rotateRight) {}

    void DoOnImageCaptured(CImageDataPointer& imageData, void* /*pUserParam*/) override {
        if (imageData->GetStatus() != GX_FRAME_STATUS_SUCCESS) return;

        try {
        int w = static_cast<int>(imageData->GetWidth());
        int h = static_cast<int>(imageData->GetHeight());
        uint64_t fid = imageData->GetFrameID();

        // 1. 先处理到局部变量（不加锁）
        cv::Mat processed(h, w, CV_8UC1);
        std::memcpy(processed.data, imageData->GetBuffer(), static_cast<size_t>(h) * w);
        if (m_sideIndex == 1 && m_rotateRight) {
            cv::Mat tmp;
            cv::rotate(processed, tmp, cv::ROTATE_180);
            processed = std::move(tmp);
        }

        // 2. 加锁写入缓冲区
        {
            std::lock_guard<std::mutex> lock(m_owner->m_bufferMutex);
            auto& buf = m_owner->m_buffers[m_sideIndex];
            buf.image = std::move(processed);
            buf.frameId = fid;
            buf.ready.store(true, std::memory_order_relaxed);
        }

        // 3. 尝试配对交付
        tryDeliver();
        } catch (...) {}
    }

private:
    void tryDeliver() {
        auto& leftBuf  = m_owner->m_buffers[0];
        auto& rightBuf = m_owner->m_buffers[1];

        // 用 CameraControl 的共享锁，保护左右两个线程互斥
        std::lock_guard<std::mutex> lock(m_owner->m_bufferMutex);

        if (!leftBuf.ready.load(std::memory_order_acquire) ||
            !rightBuf.ready.load(std::memory_order_acquire)) {
            return;
        }

        leftBuf.ready.store(false, std::memory_order_release);
        rightBuf.ready.store(false, std::memory_order_release);

        uint64_t fid = std::max(leftBuf.frameId, rightBuf.frameId);

        hal::StereoFrame frame;
        frame.frameId = fid;
        frame.timestamp = 0;
        frame.leftGray  = leftBuf.image.clone();
        frame.rightGray = rightBuf.image.clone();

        std::lock_guard cbLock(m_owner->m_callbackMutex);
        if (m_owner->m_frameCallback) {
            m_owner->m_frameCallback(frame);
        }
    }

    CameraControl* m_owner;
    int m_sideIndex;
    bool m_rotateRight;
};

// ============================================================================
// 构造 / 析构
// ============================================================================
CameraControl::CameraControl(const StereoPairConfig& config)
    : m_config(config) {}

CameraControl::~CameraControl() {
    if (m_isOpen) close();
}

// ============================================================================
// 设备信息
// ============================================================================
std::string CameraControl::getDeviceName() const { return "GalaxyStereoPair"; }
std::string CameraControl::getSerialNumber() const { return "scanner-stereo"; }

// ============================================================================
// 枚举设备
// ============================================================================
int CameraControl::enumerateDevices() {
    IGXFactory::GetInstance().Init();
    GxIAPICPP::gxdeviceinfo_vector deviceList;
    IGXFactory::GetInstance().UpdateDeviceList(1000, deviceList);
    IGXFactory::GetInstance().Uninit();

    spdlog::info("[CameraControl] 发现 {} 个设备", deviceList.size());
    for (size_t i = 0; i < deviceList.size(); ++i) {
        spdlog::info("  [{}] {} (SN: {})", i,
            (const char*)deviceList[i].GetDisplayName(),
            (const char*)deviceList[i].GetSN());
    }
    return static_cast<int>(deviceList.size());
}

// ============================================================================
// 打开 / 关闭
// ============================================================================
Result CameraControl::open() {
    if (m_isOpen) return Result::ok("设备已打开");
    const auto t0 = std::chrono::steady_clock::now();
    auto el = [t0]() { return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count(); };

    IGXFactory::GetInstance().Init();

    GxIAPICPP::gxdeviceinfo_vector deviceList;
    IGXFactory::GetInstance().UpdateDeviceList(300, deviceList);   // 枚举完成即返（原 1000ms 等满）
    spdlog::info("[CameraControl] open 计时: Init+枚举 {}ms（{} 台）", el(), deviceList.size());

    if (static_cast<int>(deviceList.size()) <= m_config.deviceIndexRight) {
        IGXFactory::GetInstance().Uninit();
        return Result::fail(-1, "设备数量不足");
    }

    for (int i = 0; i < 2; ++i) {
        int idx = (i == 0) ? m_config.deviceIndexLeft : m_config.deviceIndexRight;
        auto& side = m_sides[i];
        GxIAPICPP::gxstring sn = deviceList[idx].GetSN();
        spdlog::info("[CameraControl] 打开设备 {}: {} (SN: {})", idx,
            (const char*)deviceList[idx].GetDisplayName(), (const char*)sn);

        try {
            side.device = IGXFactory::GetInstance().OpenDeviceBySN(sn, GX_ACCESS_EXCLUSIVE);
            side.featureControl = side.device->GetRemoteFeatureControl();
            side.isOpen = true;
            spdlog::info("[CameraControl] open 计时: 设备 {} 打开完成 {}ms", idx, el());
        } catch (CGalaxyException& e) {
            spdlog::error("[CameraControl] 打开设备 {} 异常: {}", idx, e.what());
            for (int j = i - 1; j >= 0; --j) {   // 倒序关已开侧，避免半开残留
                if (m_sides[j].isOpen) {
                    m_sides[j].device->Close();
                    m_sides[j].device = CGXDevicePointer();
                    m_sides[j].featureControl = CGXFeatureControlPointer();
                    m_sides[j].isOpen = false;
                }
            }
            IGXFactory::GetInstance().Uninit();
            return Result::fail(-1, std::string("相机打开异常: ") + e.what());
        }
    }

    m_isOpen = true;
    spdlog::info("[CameraControl] 双目相机已打开");
    return Result::ok();
}

Result CameraControl::close() {
    if (!m_isOpen) return Result::ok();
    if (m_isCapturing) stopCapture();

    for (int i = 1; i >= 0; --i) {
        auto& side = m_sides[i];
        if (side.isOpen) {
            side.device->Close();
            side.device = CGXDevicePointer();
            side.featureControl = CGXFeatureControlPointer();
            side.isOpen = false;
        }
    }

    IGXFactory::GetInstance().Uninit();

    m_isOpen = false;
    spdlog::info("[CameraControl] 双目相机已关闭");
    return Result::ok();
}

bool CameraControl::isOpen() const { return m_isOpen; }

// ============================================================================
// 参数
// ============================================================================
Result CameraControl::setExposure(double ms) {
    if (!m_isOpen) return Result::fail("设备未打开");
    m_currentExposureMs = ms;
    applySideParams(0);
    applySideParams(1);
    // 读回验证
    try {
        double actual = m_sides[0].featureControl->GetFloatFeature("ExposureTime")->GetValue();
        spdlog::info("[CameraControl] 曝光设置: 请求={}ms 实际={}µs ({:.3f}ms)", ms, actual, actual/1000.0);
    } catch (...) {}
    return Result::ok();
}

void CameraControl::applySideParams(int sideIndex) {
    auto& fc = m_sides[sideIndex].featureControl;
    if (fc.IsNull()) {
        spdlog::warn("[CameraControl] applySideParams: featureControl 为空 (side={})", sideIndex);
        return;
    }
    try {
        fc->GetEnumFeature("ExposureAuto")->SetValue("Off");
        fc->GetFloatFeature("ExposureTime")->SetValue(m_currentExposureMs * 1000.0);
    } catch (CGalaxyException& e) {
        spdlog::error("[CameraControl] 曝光设置异常(side={}): {}", sideIndex, e.what());
    }
}

Result CameraControl::setGain(double dB) {
    if (!m_isOpen) return Result::fail("设备未打开");
    m_currentGain = dB;
    // 按 GainRaw 原生单位传（Galaxy GainRaw 单位非严格 dB），dB 语义由上层换算
    const int64_t raw = static_cast<int64_t>(dB);
    for (int i = 0; i < 2; ++i) {
        auto& fc = m_sides[i].featureControl;
        if (fc.IsNull()) continue;
        try {
            fc->GetEnumFeature("GainAuto")->SetValue("Off");
            fc->GetIntFeature("GainRaw")->SetValue(raw);
        } catch (CGalaxyException& e) {
            spdlog::error("[CameraControl] 增益设置异常(side={}): {}", i, e.what());
        }
    }
    // 读回验证
    try {
        int64_t actual = m_sides[0].featureControl->GetIntFeature("GainRaw")->GetValue();
        spdlog::info("[CameraControl] 增益设置: 请求={} 实际 GainRaw={}", dB, actual);
    } catch (...) {}
    return Result::ok();
}

Result CameraControl::setResolution(int width, int height) {
    if (!m_isOpen) return Result::fail("设备未打开");
    spdlog::info("[CameraControl] setResolution: 请求 {}x{}", width, height);

    for (int i = 0; i < 2; ++i) {
        auto& fc = m_sides[i].featureControl;
        if (fc.IsNull()) continue;
        try {
            // 读取传感器参数
            int64_t wMax = fc->GetIntFeature("WidthMax")->GetValue();
            int64_t hMax = fc->GetIntFeature("HeightMax")->GetValue();
            int64_t wInc = fc->GetIntFeature("Width")->GetInc();
            int64_t hInc = fc->GetIntFeature("Height")->GetInc();
            spdlog::info("[CameraControl] side={} sensor max={}x{} inc={}x{}", i, wMax, hMax, wInc, hInc);

            // 步进对齐 + 边界检查
            width = std::max((int)wInc, std::min(width, (int)wMax));
            height = std::max((int)hInc, std::min(height, (int)hMax));
            width = (width / (int)wInc) * (int)wInc;
            height = (height / (int)hInc) * (int)hInc;

            // 大恒要求: 先设 OffsetX/Y 再设 Width/Height
            int64_t offXInc = fc->GetIntFeature("OffsetX")->GetInc();
            int64_t offYInc = fc->GetIntFeature("OffsetY")->GetInc();
            int64_t offX = ((wMax - width) / 2 / offXInc) * offXInc;
            int64_t offY = ((hMax - height) / 2 / offYInc) * offYInc;

            fc->GetIntFeature("OffsetX")->SetValue(offX);
            fc->GetIntFeature("OffsetY")->SetValue(offY);
            fc->GetIntFeature("Width")->SetValue(width);
            fc->GetIntFeature("Height")->SetValue(height);

            // 读回验证
            int64_t actW = fc->GetIntFeature("Width")->GetValue();
            int64_t actH = fc->GetIntFeature("Height")->GetValue();
            spdlog::info("[CameraControl] side={} ROI 设置成功: {}x{} off=({},{})", i, actW, actH, offX, offY);

        } catch (CGalaxyException& e) {
            spdlog::error("[CameraControl] ROI 失败 side={}: {}", i, e.what());
            return Result::fail(e.what());
        }
    }
    return Result::ok();
}

// ============================================================================
// 标定（注入式缓存——B3：app 从 06 标定结果仓库喂入，08 不解析 json）
// ============================================================================
Result CameraControl::setCalibration(const hal::CameraIntrinsics& left,
                                     const hal::CameraIntrinsics& right,
                                     const hal::StereoExtrinsics& stereo) {
    std::lock_guard<std::mutex> lock(m_calibMutex);
    m_calibLeft = left;
    m_calibRight = right;
    m_calibStereo = stereo;
    return Result::ok();
}
hal::CameraIntrinsics CameraControl::getLeftIntrinsics() const {
    std::lock_guard<std::mutex> lock(m_calibMutex);
    return m_calibLeft;
}
hal::CameraIntrinsics CameraControl::getRightIntrinsics() const {
    std::lock_guard<std::mutex> lock(m_calibMutex);
    return m_calibRight;
}
hal::StereoExtrinsics CameraControl::getStereoExtrinsics() const {
    std::lock_guard<std::mutex> lock(m_calibMutex);
    return m_calibStereo;
}

// ============================================================================
// 单侧采集
// ============================================================================
void CameraControl::startSideCapture(int sideIndex) {
    auto& side = m_sides[sideIndex];

    side.stream = side.device->OpenStream(0);

    auto* handler = new CaptureEventHandler(this, sideIndex, m_config.rotateRight180);
    side.eventHandler = handler;
    side.stream->RegisterCaptureCallback(handler, nullptr);

    side.stream->StartGrab();

    applySideParams(sideIndex);

    // 解除帧率上限限制
    try {
        side.featureControl->GetBoolFeature("AcquisitionFrameRateEnable")->SetValue(false);
        spdlog::info("[CameraControl] side {} 已解除 AcquisitionFrameRate 上限", sideIndex);
    } catch (CGalaxyException&) {
        // 某些相机不支持此功能，忽略
    }

    side.featureControl->GetEnumFeature("TriggerSelector")->SetValue("FrameStart");
    side.featureControl->GetEnumFeature("TriggerMode")->SetValue("On");
    side.featureControl->GetEnumFeature("TriggerSource")->SetValue(m_config.triggerSource.c_str());
    spdlog::info("[CameraControl] side {} 硬件触发: {}", sideIndex, m_config.triggerSource);

    side.featureControl->GetCommandFeature("AcquisitionStart")->Execute();
    side.isCapturing = true;

    spdlog::info("[CameraControl] 侧 {} 采集已启动", sideIndex);
}

void CameraControl::stopSideCapture(int sideIndex) {
    auto& side = m_sides[sideIndex];
    if (!side.isCapturing) return;

    try {
        side.featureControl->GetCommandFeature("AcquisitionStop")->Execute();
        side.stream->StopGrab();
        side.stream->UnregisterCaptureCallback();
    } catch (CGalaxyException& e) {
        spdlog::error("[CameraControl] 停止采集异常: {}", e.what());
    }

    delete side.eventHandler;
    side.eventHandler = nullptr;

    side.stream->Close();
    side.stream = CGXStreamPointer();
    side.isCapturing = false;
}

// ============================================================================
// 采集控制
// ============================================================================
Result CameraControl::startCapture() {
    if (!m_isOpen) return Result::fail("设备未打开");
    if (m_isCapturing) return Result::ok("已在采集");

    try {
        startSideCapture(0);
        startSideCapture(1);
    } catch (CGalaxyException& e) {
        stopSideCapture(0);
        stopSideCapture(1);
        return Result::fail(-1, e.what());
    }

    m_isCapturing = true;
    spdlog::info("[CameraControl] 双目采集已启动");
    return Result::ok();
}

Result CameraControl::stopCapture() {
    if (!m_isCapturing) return Result::ok();

    stopSideCapture(1);
    stopSideCapture(0);

    m_isCapturing = false;
    spdlog::info("[CameraControl] 双目采集已停止");
    return Result::ok();
}

bool CameraControl::isCapturing() const { return m_isCapturing; }

// ============================================================================
// 同步抓帧
// ============================================================================
Result CameraControl::grabFrame(hal::StereoFrame& frame, int timeoutMs) {
    if (!m_isCapturing) return Result::fail("未在采集状态");

    try {
        CImageDataPointer leftData = m_sides[0].stream->GetImage(timeoutMs);
        if (leftData->GetStatus() != GX_FRAME_STATUS_SUCCESS)
            return Result::fail(-1, "左相机抓帧失败或超时");

        CImageDataPointer rightData = m_sides[1].stream->GetImage(timeoutMs);
        if (rightData->GetStatus() != GX_FRAME_STATUS_SUCCESS)
            return Result::fail(-2, "右相机抓帧失败或超时");

        frame.frameId = leftData->GetFrameID();
        frame.timestamp = leftData->GetTimeStamp();

        int lw = static_cast<int>(leftData->GetWidth());
        int lh = static_cast<int>(leftData->GetHeight());
        frame.leftGray.create(lh, lw, CV_8UC1);
        std::memcpy(frame.leftGray.data, leftData->GetBuffer(), static_cast<size_t>(lh) * lw);

        int rw = static_cast<int>(rightData->GetWidth());
        int rh = static_cast<int>(rightData->GetHeight());
        cv::Mat temp(rh, rw, CV_8UC1);
        std::memcpy(temp.data, rightData->GetBuffer(), static_cast<size_t>(rh) * rw);

        if (m_config.rotateRight180)
            cv::rotate(temp, frame.rightGray, cv::ROTATE_180);
        else
            frame.rightGray = std::move(temp);

        return Result::ok();
    } catch (CGalaxyException& e) {
        return Result::fail(-3, e.what());
    }
}

// ============================================================================
// 异步采集
// ============================================================================
Result CameraControl::startAsyncCapture(hal::FrameCallback cb) {
    {
        std::lock_guard lock(m_callbackMutex);
        m_frameCallback = std::move(cb);
    }
    return startCapture();
}

Result CameraControl::stopAsyncCapture() {
    auto r = stopCapture();
    {
        std::lock_guard lock(m_callbackMutex);
        m_frameCallback = nullptr;
    }
    return r;
}

// ============================================================================
// 温度
// ============================================================================
double CameraControl::getTemperature() const {
    if (!m_isOpen) return 0.0;
    try {
        auto& fc = const_cast<CGXFeatureControlPointer&>(m_sides[0].featureControl);
        if (fc.IsNull()) return 0.0;
        auto feature = fc->GetFloatFeature("DeviceTemperature");
        if (feature.IsNull()) return 0.0;
        return feature->GetValue();
    } catch (CGalaxyException&) {
        return 0.0;
    }
}

} // namespace Scanner::device
