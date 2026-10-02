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
        if (imageData->GetStatus() != GX_FRAME_STATUS_SUCCESS) {
            // 坏帧可见性（260926）：链路错误/传输不完整帧此前静默吞——计数＋首帧/每
            // 100 帧记一条（status 对照 GX_FRAME_STATUS_LIST；与帧号回退同源＝USB
            // 拥塞指纹——双目全分辨率带宽顶格时链路层出错重开）
            try {
                const uint64_t n = s_badFrames[static_cast<size_t>(m_sideIndex)]
                                       .fetch_add(1, std::memory_order_relaxed) + 1;
                if (n == 1 || n % 100 == 0) {
                    JMW_LOG_WARN("08-CameraControl",
                        "[CameraControl] 坏帧丢弃（side={} 累计 {}，status={}，fid={}）"
                        "——USB 链路错误/传输不完整指纹",
                        m_sideIndex, n, static_cast<int>(imageData->GetStatus()),
                        imageData->GetFrameID());
                }
            } catch (...) {}
            return;
        }

        try {
        int w = static_cast<int>(imageData->GetWidth());
        int h = static_cast<int>(imageData->GetHeight());
        uint64_t fid = imageData->GetFrameID();
        const uint64_t ts = imageData->GetTimeStamp();

        // 时间戳配对·周期估计（260927 方案B）：相邻帧 ts 差≈触发周期——ε=period/4
        // 的单位无关基准；重开首帧跨零/巨跳（>2^40）不入账防污染
        if (m_lastTs != 0 && ts > m_lastTs) {
            const uint64_t d = ts - m_lastTs;
            if (d < (1ull << 40))
                m_owner->m_sideTsPeriod[m_sideIndex].store(d, std::memory_order_relaxed);
        }
        m_lastTs = ts;

        // 帧号回退检测（260926）：同侧帧号不升反降＝相机/USB 链路层自发重开出流
        //（BlockID 归零——app 未重启流）。真机实证：左右轮流回跳 1~6、另一侧正常
        // 爬升，为配对丢组＋偏移翻转采纳的根因指纹（全分辨率×50Hz×双目≈3.1Gbps
        // 顶格 USB3 单控制器实用上限）
        if (m_lastFid != 0 && fid < m_lastFid) {
            s_rollbacks[static_cast<size_t>(m_sideIndex)]
                .fetch_add(1, std::memory_order_relaxed);   // 回退累计出口（自动恢复检测源）
            JMW_LOG_WARN("08-CameraControl",
                "[CameraControl] 帧号回退（side={}：{}→{}）——相机/USB 链路层重开出流"
                "指纹（查带宽/USB 控制器分配/线缆）",
                m_sideIndex, m_lastFid, fid);
        }
        m_lastFid = fid;

        // 1. 先处理到局部变量（不加锁）
        cv::Mat processed(h, w, CV_8UC1);
        // 对比度·左右分置（260927→1002 换自建 LUT）：SDK ImageImprovment 对 Mono8
        // 实测静默 no-op（返回 null 不变换）——改用 OpenCV cv::LUT 逐像素真实对比度
        // 变换（教科书公式 output=(input−128)×factor+128，factor=1+ctr/100）。
        // 0=直通零开销；LUT 256 项懒建/值变重建（仅本侧回调线程触碰 side 资源）
        const int ctr = (m_sideIndex == 0)
            ? m_owner->m_contrastL.load(std::memory_order_relaxed)
            : m_owner->m_contrastR.load(std::memory_order_relaxed);
        if (ctr != 0) {
            auto& side = m_owner->m_sides[m_sideIndex];
            if (side.lutContrast != ctr || side.contrastLut.empty()) {
                const double factor = 1.0 + ctr / 100.0;
                side.contrastLut = cv::Mat(1, 256, CV_8UC1);
                for (int i = 0; i < 256; ++i) {
                    const int v = static_cast<int>((i - 128) * factor + 128 + 0.5);
                    side.contrastLut.at<uint8_t>(0, i) =
                        static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
                }
                side.lutContrast = ctr;
            }
            // 源帧建 Mat 头（零拷贝）→ LUT 变换到 processed（一次遍历＝拷贝＋对比度）
            cv::Mat srcMat(h, w, CV_8UC1, const_cast<void*>(imageData->GetBuffer()));
            cv::LUT(srcMat, side.contrastLut, processed);
            // 一次性数值探针（260927）：原始 vs 变换后前 4 像素——数学铁证
            {
                static std::atomic<uint64_t> s_lutProbe{0};
                if (s_lutProbe.fetch_add(1, std::memory_order_relaxed) < 2) {
                    const auto* raw = static_cast<const uint8_t*>(imageData->GetBuffer());
                    JMW_LOG_INFO("08-CameraControl",
                        "[CameraControl] LUT 探针 side={} ctr={} 原始[0..3]={},{},{},{} "
                        "→ 变换后={},{},{},{}",
                        m_sideIndex, ctr,
                        raw[0], raw[1], raw[2], raw[3],
                        processed.data[0], processed.data[1],
                        processed.data[2], processed.data[3]);
                }
            }
        } else {
            std::memcpy(processed.data, imageData->GetBuffer(), static_cast<size_t>(h) * w);
        }
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
            buf.timestamp = ts;
            buf.ready.store(true, std::memory_order_relaxed);
        }

        // 3. 尝试配对交付
        tryDeliver();
        } catch (const std::exception& e) {
            // 回调异常可见性（260926）：此前 catch(...) 静默吞——SDK 回调线程里任何
            // 异常都意味着丢帧且无痕
            JMW_LOG_ERROR("08-CameraControl",
                "[CameraControl] 采集回调异常（side={}）: {}", m_sideIndex, e.what());
        } catch (...) {
            JMW_LOG_ERROR("08-CameraControl",
                "[CameraControl] 采集回调异常（side={}）: 未知类型", m_sideIndex);
        }
    }

public:
    // 帧号回退计数（per-side、跨会话累计）——DeviceManager 链路自动恢复检测源
    //（260927 方案A实验；跨会话不归零与 s_badFrames/s_mismatchDrops 同口径——消费侧记基线）
    inline static std::atomic<uint64_t> s_rollbacks[2]{};

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

        // —— 配对判组（260927 用户定版：时间戳直接配对）——
        // 方案B 主判据：Δ（两机钟差）已锚定即以 |L.ts−R.ts−Δ| < ε=周期/4 为唯一
        // 同触发判据——帧号不参与判组（只作信息随帧/回退检测源）。真机实证
        //（260927 日志）：时间戳不随流重开复位（三次重开 Δ 变化仅 ~24µs≈3ppm
        // 晶振漂移级），链路重开/漏触发后零中断续配；d>ε=L 超前丢 R、d<−ε=R
        // 超前丢 L（被丢帧的伙伴已物理丢失，等不回）。ε 用 min(周期L,周期R)——
        // 单侧重启间隙会污染本侧周期估计（实测 266ms 混入），取健康侧为准。
        // Δ 锚定前（新开流首对）与开关关闭时走下方案遗留路径：帧号严格等值
        //（2026-09-05 口径）＋稳定偏移采纳（260911）——其严格等值交付同时锚定 Δ
        //（此时新开流两侧帧号同步起步，等值对即同触发对，锚定可靠）。
        // 260926 实验开关 pairStrictFrameId=false：遗留路径整段旁路——双方 ready
        // 即按时间对齐交付（测帧率用；同刻性/T-V 归属不保证，仅实验）
        const bool tsDirect = m_owner->m_config.timestampPairing && m_owner->m_pairTsValid;
        if (tsDirect) {
            const uint64_t pl = m_owner->m_sideTsPeriod[0].load(std::memory_order_relaxed);
            const uint64_t pr = m_owner->m_sideTsPeriod[1].load(std::memory_order_relaxed);
            const uint64_t period = pl < pr ? pl : pr;   // min：健康侧周期为准
            const int64_t d = static_cast<int64_t>(leftBuf.timestamp) -
                              static_cast<int64_t>(rightBuf.timestamp) -
                              m_owner->m_pairTsDelta;
            const int64_t eps = static_cast<int64_t>(period / 4);
            const bool hit = d < eps && d > -eps;
            static std::atomic<uint64_t> s_tsStats{0};   // 直配可观测性（首对+每 200 对）
            if (!hit || s_tsStats.fetch_add(1, std::memory_order_relaxed) % 200 == 0) {
                JMW_LOG_INFO("08-CameraControl",
                    "[CameraControl] 时间戳直配{}：L.fid={} R.fid={} d={}ns ε={}ns Δ={}ns"
                    "（{}）",
                    hit ? "命中" : "未命中丢旧", leftBuf.frameId, rightBuf.frameId,
                    d, eps, m_owner->m_pairTsDelta,
                    hit ? "" : "——被丢帧的伙伴已物理丢失");
            }
            if (!hit) {
                auto& stale = (d >= eps) ? rightBuf : leftBuf;   // 超前侧保留等下一伙伴
                stale.image.release();
                stale.ready.store(false, std::memory_order_release);
                return;
            }
        } else if (m_owner->m_config.pairStrictFrameId && leftBuf.frameId != rightBuf.frameId) {
            // —— 遗留路径（Δ 未锚定/方案B关闭）：帧号严格等值＋偏移采纳 ——
            //（历史实证：模式切换/调参 N10 重发后某侧多吃漏吃一触发沿→L-R 恒差
            // ±1 永不收敛，严格等值=每帧全丢——连续 8 次同号失配即采纳偏移按
            // 时间对齐交付；T/V 奇偶归属可能有疑——WARN 留痕）
            const int64_t off = static_cast<int64_t>(leftBuf.frameId) -
                                static_cast<int64_t>(rightBuf.frameId);
            if (off != m_owner->m_pairOffset) {
                if (off == m_owner->m_lastMismatchOff) {
                    if (++m_owner->m_mismatchStreak >= 8) {
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

        // —— 方案B·Δ 学习：直配模式所有交付对喂（命中对 d 天然小，EMA 稳收敛且
        //    跟踪晶拖）；遗留路径仅严格等值对喂（偏移采纳对可能跨触发，防错误
        //    锚定）。首对锚定，其后 EMA α=1/16——
        if (tsDirect || leftBuf.frameId == rightBuf.frameId) {
            const int64_t d = static_cast<int64_t>(leftBuf.timestamp) -
                              static_cast<int64_t>(rightBuf.timestamp);
            m_owner->m_pairTsDelta = m_owner->m_pairTsValid
                ? m_owner->m_pairTsDelta + (d - m_owner->m_pairTsDelta) / 16
                : d;
            m_owner->m_pairTsValid = true;
        }

        hal::StereoFrame frame;
        frame.frameId = leftBuf.frameId;   // 严格配对下左右相等（偏移配对=左号为准）
        frame.frameIdLeft = leftBuf.frameId;    // 原始帧号（调试显示）
        frame.frameIdRight = rightBuf.frameId;
        frame.timestamp = 0;
        frame.timestampLeft = leftBuf.timestamp;    // 设备原始时间戳（方案B 可观测性）
        frame.timestampRight = rightBuf.timestamp;
        // —— T/V 激光组判定（260927 时间戳奇偶法）：脉冲序＝round((tsL−t₀)/周期)
        //    的奇偶（偶=T 左斜）；t₀ 未锚定（开流首对）在本对锚定并判 T。周期用
        //    min(双侧)（健康侧为准）；周期未知时本帧不判（tvKnown=false——消费方
        //    回退帧号奇偶），锚照常立（下帧起可判）——
        {
            const uint64_t pl = m_owner->m_sideTsPeriod[0].load(std::memory_order_relaxed);
            const uint64_t pr = m_owner->m_sideTsPeriod[1].load(std::memory_order_relaxed);
            const uint64_t tvPeriod = (pl && pr) ? (pl < pr ? pl : pr) : 0;
            if (!m_owner->m_tvT0Valid) {
                m_owner->m_tvT0 = leftBuf.timestamp;
                m_owner->m_tvT0Valid = true;
            }
            if (tvPeriod > 0 && leftBuf.timestamp >= m_owner->m_tvT0) {
                const uint64_t idx =
                    (leftBuf.timestamp - m_owner->m_tvT0 + tvPeriod / 2) / tvPeriod;
                frame.tvKnown = true;
                frame.tvLeftSkew = (idx % 2) == 0;   // 偶脉冲=T 左斜（锚=首脉冲 T）
            }
        }
        frame.leftGray  = leftBuf.image.clone();
        frame.rightGray = rightBuf.image.clone();

        // 开流/复采后首组可见性（260927 方案A实验）：自动恢复编排的成活判据——
        // 严格等值首组 L==R 且从 0 起步＝双流重开对齐成功；偏移配对首组＝有侧漏
        // 首触发（T/V 相位存疑，需复核）。附带方案B 基线值（Δ/周期）供判读
        if (m_owner->m_firstPairPending.exchange(false, std::memory_order_acq_rel)) {
            JMW_LOG_INFO("08-CameraControl",
                "[CameraControl] 开流后首组：L={} R={}（{}配对，采纳偏移={}）"
                " tsΔ={} 周期L/R={}/{}",
                leftBuf.frameId, rightBuf.frameId,
                leftBuf.frameId == rightBuf.frameId ? "严格等值" : "偏移",
                m_owner->m_pairOffset,
                m_owner->m_pairTsValid ? m_owner->m_pairTsDelta : 0,
                m_owner->m_sideTsPeriod[0].load(std::memory_order_relaxed),
                m_owner->m_sideTsPeriod[1].load(std::memory_order_relaxed));
        }

        std::lock_guard cbLock(m_owner->m_callbackMutex);
        if (m_owner->m_frameCallback) {
            m_owner->m_frameCallback(frame);
        }
    }

    CameraControl* m_owner;
    int m_sideIndex;
    bool m_rotateRight;
    // 坏帧计数（per-side、跨会话累计——与 tryDeliver 内 s_mismatchDrops 同口径）
    inline static std::atomic<uint64_t> s_badFrames[2]{};
    uint64_t m_lastFid = 0;   // 本侧最近帧号（每侧回调单线程——无锁；帧号回退检测基线，
                              // 新会话随 handler 重建归零）
    uint64_t m_lastTs = 0;    // 本侧最近时间戳（周期估计用——同上单线程无锁）
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

    // 枚举重试（260927 真机实证：快速重启时前一实例刚退出、USB 相机未及重枚举
    // ——首枚举 0 台即失败＝自检全链红根因）。0 台时 1.5s 间隔重试，总窗 ~8s
    GxIAPICPP::gxdeviceinfo_vector deviceList;
    for (int attempt = 1; ; ++attempt) {
        deviceList.clear();
        IGXFactory::GetInstance().UpdateDeviceList(300, deviceList);   // 枚举超时减半（原 1000ms 起步快）
        JMW_LOG_INFO("08-CameraControl",
            "[CameraControl] open 计时: Init+枚举 {}ms（{} 台，第 {} 次）",
            el(), deviceList.size(), attempt);
        if (static_cast<int>(deviceList.size()) > m_config.deviceIndexRight) break;
        if (attempt >= 6) {
            IGXFactory::GetInstance().Uninit();
            return Result::fail(-1, "设备枚举不足（重试 6 次仍 0 台——检查相机 USB/"
                                    "是否残留实例占用，重插后重启）");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
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
    // applySideParams 重试后仍败会上抛（开流路径语义）——参数路径就地收口为
    // Result fail（逻辑线程不扩散异常）
    try {
        applySideParams(0);
        applySideParams(1);
    } catch (CGalaxyException& e) {
        return Result::fail(e.what());
    }
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
    // —— 260912 假活根因收口：停→快启窗口 USB 控制通道瞬时拥塞（-1010 TL 0x16）
    //    原实现吞异常＝带病启动（状态已启动/0 帧/无报错——预览死真机实证）。
    //    有界重试 3×200ms（同 AcquisitionStop 口径）；仍败上抛——startCapture
    //    的整设备复位重开路径接管自愈
    for (int attempt = 1; ; ++attempt) {
        try {
            fc->GetEnumFeature("ExposureAuto")->SetValue("Off");
            fc->GetFloatFeature("ExposureTime")->SetValue(m_currentExposureMs * 1000.0);
            return;
        } catch (CGalaxyException& e) {
            if (attempt >= 3) {
                JMW_LOG_ERROR("08-CameraControl",
                    "[CameraControl] 曝光设置 3 次均败(side={}): {}——上抛触发复位重开",
                    sideIndex, e.what());
                throw;
            }
            JMW_LOG_WARN("08-CameraControl",
                "[CameraControl] 曝光设置失败（第 {} 次，side={}）: {}——200ms 后重试",
                attempt, sideIndex, e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
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

// 软件端对比度·左右分置（260927→1002）：仅记账原子值——逐帧应用在采集回调
//（ImageProcess 懒建配置，值变重建）；0=直通零开销。设备端无对比度特性
//（SDK 官方口径：软件图像增强 SetContrastParam，0=不变/>0 增强/<0 减弱）
Result CameraControl::setContrast(int leftValue, int rightValue) {
    const auto clamp = [](int v) { return v < -100 ? -100 : (v > 100 ? 100 : v); };
    const int lv = clamp(leftValue);
    const int rv = clamp(rightValue);
    m_contrastL.store(lv, std::memory_order_relaxed);
    m_contrastR.store(rv, std::memory_order_relaxed);
    JMW_LOG_INFO("08-CameraControl",
        "[CameraControl] 对比度设置·左右分置: L={} R={}（0=直通；逐帧软件增强）", lv, rv);
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
        m_pairTsValid = false;          // 方案B：钟差随新开流重锚（首严格等值对起）
        m_tvT0Valid = false;            // T/V 锚随新开流重立（首交付对=T 左斜）
    }
    m_firstPairPending.store(true, std::memory_order_release);   // 首组日志武装（260927）
    if (!m_config.pairStrictFrameId) {
        JMW_LOG_WARN("08-CameraControl",
            "[CameraControl] 帧号严格配对已关（pairStrictFrameId=false）——按时间对齐"
            "交付：仅限带宽/帧率实测（T/V 奇偶归属与立体同刻性不保证，数据不可用于扫描）");
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

// 帧号回退累计（双侧求和；跨会话单调——消费侧 DeviceManager 记基线取增量）
uint64_t CameraControl::frameRollbackCount() const {
    return CaptureEventHandler::s_rollbacks[0].load(std::memory_order_relaxed) +
           CaptureEventHandler::s_rollbacks[1].load(std::memory_order_relaxed);
}

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
