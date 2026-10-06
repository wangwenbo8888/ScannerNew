#pragma once
// ============================================================================
// ScannerWindow.h — 扫描仪控制窗口（基于 LEADSCANSeries.ui）
//
// A-T17 串口旁路收口：设备（相机/下位机）一律经 AppContext→DeviceManager 门面，
// 本窗口不再私摸 QSerialPort/CameraControl（08 红线）。
// ============================================================================

#include <QMainWindow>
#include <QTimer>
#include <QSlider>     // P-5 亮度档滑条
#include <QLabel>
#include <atomic>
#include <mutex>
#include <thread>
#include "ui_ScannerWindow.h"

#include "base/types.h"
#include "modules/DeviceManager/IScannerCamera.h"   // StereoFrame（帧出口签名）
#include "FrameBuffer.h"

class AppContext;

class ScannerWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit ScannerWindow(AppContext* appCtx = nullptr, QWidget *parent = nullptr);
    ~ScannerWindow();

    uint64_t getFrameCount() const { return m_frameCount; }
    uint64_t getCurrentFps() const { return m_currentFps; }

private slots:
    void onOpenScannerCamera();
    void onCloseScannerCamera();
    void onStartScanner();
    void onStopScanner();
    void onCalibrateClicked();

    void onSliderBackgroundChanged(int v);   // 261002：频率滑条移除（freqHz 固定 60 无入口）
    void onSliderLaserChanged(int v);
    void onSliderExposeChanged(int v);
    void onResolutionChanged(int index);

    void onTimeout();

signals:
    void updateImages(QImage left, QImage right);

private:
    Ui::LEADSCANSeriesClass ui;

    AppContext* m_appCtx = nullptr;
    Scanner::data::FrameBuffer* m_frameBuffer = nullptr;
    QTimer* m_fpsTimer = nullptr;
    QTimer* m_consumerTimer = nullptr;
    QComboBox* m_resCombo = nullptr;

    // —— P-5 亮度档滑条收敛（261002 §3.4）：三参滑条退役，换 20 档预设档滑条 ——
    QSlider* m_ladderSlider = nullptr;   // 亮度档滑条（与按键同梯同账，松手提交）
    QLabel*  m_ladderLbl = nullptr;      // 档号回显 "N/20"
    QLabel*  m_ladderRO = nullptr;       // 三参只读提示行

    uint64_t m_frameCount = 0;
    uint64_t m_prevFrameCount = 0;
    uint64_t m_currentFps = 0;

    // P3 渲染加固：处理帧率估算器（HardwareMonitor 巡检线程调——线程安全差分）
    struct ProcFpsEstimator {
        mutable std::mutex mtx;
        uint64_t lastCount = 0;
        int64_t lastMs = 0;
        double estimate(uint64_t count);
    } m_procFps;

    // 设备开/关移后台线程（dm->open/close 含自动搜口+join 逻辑线程+Galaxy 关闭，
    // 同步跑在 UI 线程会冻窗 ~2s——观感即"按钮没作用"）。busy 期间四个设备按钮禁用。
    std::thread m_devThread;
    std::atomic<bool> m_devBusy{false};
    bool startDeviceOp();               // UI 线程：busy 守卫+禁按钮；false=已在操作中
    void endDeviceOp();                 // UI 线程：收线程+复按钮（设备回调尾调）
    void setDeviceButtonsEnabled(bool on);
    void resetPreviewUi(bool stopConsumer);  // 预览复位：清残影+帧率清零（stopConsumer=连显示定时器停）

    // 帧出口（DeviceManager::startFrameStream 注册——计数+入队，最轻量）
    void pushFrameToBuffer(const Scanner::hal::StereoFrame& frame);
};
