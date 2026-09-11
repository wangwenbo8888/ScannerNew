// ============================================================================
// CameraControl.cpp — 大恒 Galaxy SDK 相机采集实现
// ============================================================================

#include "CameraControl.h"
#include <spdlog/spdlog.h>
#include "jmw_logging.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

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

        // —— 帧号配对（用户口径 2026-09-05 严格等值；260911 补稳定偏移采纳）——
        // 严格等值保奇偶分派（激光 T/V 左斜/右斜组别）与立体同刻性；不等则丢
        // 落后侧图像（保留超前侧待追平）。**实机实证（260911 日志）**：模式切换/
        // 调参触发 N10 重发后，某侧多吃/漏吃一个触发沿→L-R 恒差 ±1 且永不收敛
        // ——严格等值=每帧全丢（预览冻结/零数据，双相机实都在出流）。故补：
        // 连续 30 次同号失配（60Hz 下 ~0.5s）即采纳该偏移，按时间对齐配对交付；
        // 激光 T/V 奇偶归属可能有疑（无法从帧号判定哪侧失步）——采纳时 WARN
        // 大声告警，frameIdLeft/Right 原始帧号照带上报供监视窗「偏移」显示。
        int64_t off = 0;
        if (leftBuf.frameId != rightBuf.frameId) {
            off = static_cast<int64_t>(leftBuf.frameId) -
                  static_cast<int64_t>(rightBuf.frameId);
            // 偏移估计器（恒跑：失配≠已采纳偏移时计数；漂移后同样可再收敛）
            if (off != m_owner->m_pairOffset) {
                if (off == m_owner->m_lastMismatchOff) {
                    if (++m_owner->m_mismatchStreak >= 30) {
                        const int64_t old = m_owner->m_pairOffset;
                        m_owner->m_pairOffset = off;
                        JMW_LOG_WARN("08-CameraControl",
                            "[CameraControl] 帧号稳定偏移 {}→{} 采纳（L-R 恒差 {}："
                            "按时间对齐配对——激光 T/V 奇偶归属可能有疑，扫描数据"
                            "需复核；建议稍后重启采集复位偏移）",
                            old, off, off);
                        m_owner->m_mismatchStreak = 0;
                    }
                } else {
                    m_owner->m_lastMismatchOff = off;
                    m_owner->m_mismatchStreak = 1;
                }
            }
            if (off != m_owner->m_pairOffset) {
                auto& behind = (leftBuf.frameId < rightBuf.frameId) ? leftBuf : rightBuf;
                behind.image.release();
                behind.ready.store(false, std::memory_order_release);
                static std::atomic<uint64_t> s_mismatchDrops{0};
                if (s_mismatchDrops.fetch_add(1) % 100 == 0) {
                    JMW_LOG_WARN("08-CameraControl",
                                 "[CameraControl] 帧号不配丢组（累计 {}）：L={} R={}（采纳偏移={}）",
                                 s_mismatchDrops.load(), leftBuf.frameId, rightBuf.frameId,
                                 m_owner->m_pairOffset);
                }
                return;
            }
            // off == pairOffset：按稳定偏移配对（落入下方交付）
        }

        leftBuf.ready.store(false, std::memory_order_release);
        rightBuf.ready.store(false, std::memory_order_release);

        hal::StereoFrame frame;
        frame.frameId = leftBuf.frameId;   // 严格配对下左右相等（偏移配对=左号为准）
        frame.frameIdLeft = leftBuf.frameId;    // 原始帧号（调试显示）
        frame.frameIdRight = rightBuf.frameId;
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

    JMW_LOG_INFO("08-CameraControl", "[CameraControl] 发现 {} 个设备", deviceList.size());
    for (size_t i = 0; i < deviceList.size(); ++i) {
        JMW_LOG_INFO("08-CameraControl", "  [{}] {} (SN: {})", i,
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
    JMW_LOG_INFO("08-CameraControl", "[CameraControl] open 计时: Init+枚举 {}ms（{} 台）", el(), deviceList.size());

    if (static_cast<int>(deviceList.size()) <= m_config.deviceIndexRight) {
        IGXFactory::GetInstance().Uninit();
        return Result::fail(-1, "设备数量不足");
    }

    for (int i = 0; i < 2; ++i) {
        int idx = (i == 0) ? m_config.deviceIndexLeft : m_config.deviceIndexRight;
        auto& side = m_sides[i];
        GxIAPICPP::gxstring sn = deviceList[idx].GetSN();
        JMW_LOG_INFO("08-CameraControl", "[CameraControl] 打开设备 {}: {} (SN: {})", idx,
            (const char*)deviceList[idx].GetDisplayName(), (const char*)sn);

        try {
            side.device = IGXFactory::GetInstance().OpenDeviceBySN(sn, GX_ACCESS_EXCLUSIVE);
            side.featureControl = side.device->GetRemoteFeatureControl();
            side.isOpen = true;
            JMW_LOG_INFO("08-CameraControl", "[CameraControl] open 计时: 设备 {} 打开完成 {}ms", idx, el());
        } catch (CGalaxyException& e) {
            JMW_LOG_ERROR("08-CameraControl", "[CameraControl] 打开设备 {} 异常: {}", idx, e.what());
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
    JMW_LOG_INFO("08-CameraControl", "[CameraControl] 双目相机已打开");
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
    JMW_LOG_INFO("08-CameraControl", "[CameraControl] 双目相机已关闭");
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
        JMW_LOG_INFO("08-CameraControl", "[CameraControl] 曝光设置: 请求={}ms 实际={}µs ({:.3f}ms)", ms, actual, actual/1000.0);
    } catch (...) {}
    return Result::ok();
}

void CameraControl::applySideParams(int sideIndex) {
    auto& fc = m_sides[sideIndex].featureControl;
    if (fc.IsNull()) {
        JMW_LOG_WARN("08-CameraControl", "[CameraControl] applySideParams: featureControl 为空 (side={})", sideIndex);
        return;
    }
    try {
        fc->GetEnumFeature("ExposureAuto")->SetValue("Off");
        fc->GetFloatFeature("ExposureTime")->SetValue(m_currentExposureMs * 1000.0);
    } catch (CGalaxyException& e) {
        JMW_LOG_ERROR("08-CameraControl", "[CameraControl] 曝光设置异常(side={}): {}", sideIndex, e.what());
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
            JMW_LOG_ERROR("08-CameraControl", "[CameraControl] 增益设置异常(side={}): {}", i, e.what());
        }
    }
    // 读回验证
    try {
        int64_t actual = m_sides[0].featureControl->GetIntFeature("GainRaw")->GetValue();
        JMW_LOG_INFO("08-CameraControl", "[CameraControl] 增益设置: 请求={} 实际 GainRaw={}", dB, actual);
    } catch (...) {}
    return Result::ok();
}

Result CameraControl::setResolution(int width, int height) {
    if (!m_isOpen) return Result::fail("设备未打开");
    JMW_LOG_INFO("08-CameraControl", "[CameraControl] setResolution: 请求 {}x{}", width, height);

    for (int i = 0; i < 2; ++i) {
        auto& fc = m_sides[i].featureControl;
        if (fc.IsNull()) continue;
        try {
            // 读取传感器参数
            int64_t wMax = fc->GetIntFeature("WidthMax")->GetValue();
            int64_t hMax = fc->GetIntFeature("HeightMax")->GetValue();
            int64_t wInc = fc->GetIntFeature("Width")->GetInc();
            int64_t hInc = fc->GetIntFeature("Height")->GetInc();
            JMW_LOG_INFO("08-CameraControl", "[CameraControl] side={} sensor max={}x{} inc={}x{}", i, wMax, hMax, wInc, hInc);

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
            JMW_LOG_INFO("08-CameraControl", "[CameraControl] side={} ROI 设置成功: {}x{} off=({},{})", i, actW, actH, offX, offY);

        } catch (CGalaxyException& e) {
            JMW_LOG_ERROR("08-CameraControl", "[CameraControl] ROI 失败 side={}: {}", i, e.what());
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
        JMW_LOG_INFO("08-CameraControl", "[CameraControl] side {} 已解除 AcquisitionFrameRate 上限", sideIndex);
    } catch (CGalaxyException&) {
        // 某些相机不支持此功能，忽略
    }

    side.featureControl->GetEnumFeature("TriggerSelector")->SetValue("FrameStart");
    side.featureControl->GetEnumFeature("TriggerMode")->SetValue("On");
    side.featureControl->GetEnumFeature("TriggerSource")->SetValue(m_config.triggerSource.c_str());
    JMW_LOG_INFO("08-CameraControl", "[CameraControl] side {} 硬件触发: {}", sideIndex, m_config.triggerSource);

    side.featureControl->GetCommandFeature("AcquisitionStart")->Execute();
    side.isCapturing = true;

    JMW_LOG_INFO("08-CameraControl", "[CameraControl] 侧 {} 采集已启动", sideIndex);
}

bool CameraControl::stopSideCapture(int sideIndex) {
    auto& side = m_sides[sideIndex];
    if (!side.isCapturing) return true;

    // —— 260911 停启楔死根因收口（真机日志实证）——
    // 原序 AcquisitionStop→StopGrab→Unregister 单 try：AcquisitionStop 瞬时 USB
    // 拥塞失败（-1010 TL 0x16）即整段跳过——设备滞留「采集中」态且 SDK 持已删
    // 回调，此后 OpenStream 恒败（停→启预览冻结）。现拆分防护：
    // ① StopGrab 先行——停帧投递降 USB 压力，后续控制命令更易成功
    // ② AcquisitionStop 有界重试（3 次×200ms）——瞬时拥塞可自愈
    // ③ UnregisterCaptureCallback 恒在 delete 前（异常路径不跳过）
    try {
        side.stream->StopGrab();
    } catch (CGalaxyException& e) {
        JMW_LOG_WARN("08-CameraControl", "[CameraControl] StopGrab 异常(side={}): {}",
                     sideIndex, e.what());
    }
    bool acqStopped = false;
    for (int attempt = 1; attempt <= 3; ++attempt) {
        try {
            side.featureControl->GetCommandFeature("AcquisitionStop")->Execute();
            acqStopped = true;
            break;
        } catch (CGalaxyException& e) {
            JMW_LOG_WARN("08-CameraControl",
                "[CameraControl] AcquisitionStop 失败（第 {} 次，side={}）: {}",
                attempt, sideIndex, e.what());
            if (attempt < 3) std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }
    if (!acqStopped) {
        // 设备滞留采集态——上层 startCapture 的复位重开路径可救（此处仅记录；
        // Fault 归 DeviceManager 收口）
        JMW_LOG_ERROR("08-CameraControl",
            "[CameraControl] AcquisitionStop 三次均败（side={}）——设备滞留采集态，"
            "下次开流将触发整设备复位重开", sideIndex);
    }
    try {
        side.stream->UnregisterCaptureCallback();
    } catch (CGalaxyException& e) {
        JMW_LOG_WARN("08-CameraControl", "[CameraControl] 注销回调异常(side={}): {}",
                     sideIndex, e.what());
    }

    delete side.eventHandler;
    side.eventHandler = nullptr;

    try {
        side.stream->Close();
    } catch (CGalaxyException& e) {
        JMW_LOG_WARN("08-CameraControl", "[CameraControl] 关流异常(side={}): {}",
                     sideIndex, e.what());
    }
    side.stream = CGXStreamPointer();
    side.isCapturing = false;
    return acqStopped;
}

// ============================================================================
// 采集控制
// ============================================================================
Result CameraControl::startCapture() {
    if (!m_isOpen) return Result::fail("设备未打开");
    if (m_isCapturing) return Result::ok("已在采集");

    // 偏移配对状态复位（新开流帧号重新起步——旧偏移不作数；此时无回调并发）
    {
        std::lock_guard<std::mutex> lock(m_bufferMutex);
        m_pairOffset = 0;
        m_lastMismatchOff = 0;
        m_mismatchStreak = 0;
    }

    try {
        startSideCapture(0);
        startSideCapture(1);
    } catch (CGalaxyException& e) {
        stopSideCapture(0);
        stopSideCapture(1);
        // —— 260911 楔死自愈：开流 USB 失败（TL 0x16）多为设备滞留采集态（停侧
        //    AcquisitionStop 曾失败）——原地重试无解；整设备 close→open（USB 重
        //    枚举复位传输通道）后重试一次，等价人工"关相机再开"恢复路径
        JMW_LOG_WARN("08-CameraControl",
                     "[CameraControl] 开流失败（{}）——整设备复位重开后重试", e.what());
        close();
        const auto ro = open();
        if (!ro.success)
            return Result::fail(-1, std::string("相机复位重开失败: ") + ro.message);
        try {
            startSideCapture(0);
            startSideCapture(1);
        } catch (CGalaxyException& e2) {
            stopSideCapture(0);
            stopSideCapture(1);
            return Result::fail(-1, std::string("复位后开流仍失败: ") + e2.what());
        }
    }

    m_isCapturing = true;
    JMW_LOG_INFO("08-CameraControl", "[CameraControl] 双目采集已启动");
    return Result::ok();
}

Result CameraControl::stopCapture() {
    if (!m_isCapturing) return Result::ok();

    const bool clean1 = stopSideCapture(1);
    const bool clean0 = stopSideCapture(0);

    m_isCapturing = false;
    JMW_LOG_INFO("08-CameraControl", "[CameraControl] 双目采集已停止");
    // 停侧不净（AcquisitionStop 重试仍败——设备滞留采集态）：报失败给上层出
    // Fault；下次 startCapture 的复位重开路径自愈
    if (!clean0 || !clean1)
        return Result::fail(-1, "AcquisitionStop 未干净收口（设备滞留采集态——"
                                "下次开流将整设备复位重开）");
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
