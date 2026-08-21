#pragma once
// ============================================================================
// ScannerWindow.h — 扫描仪控制窗口（基于 LEADSCANSeries.ui）
//
// A-T17 串口旁路收口：设备（相机/下位机）一律经 AppContext→DeviceManager 门面，
// 本窗口不再私摸 QSerialPort/CameraControl（08 红线）。
// ============================================================================

#include <QMainWindow>
#include <QTimer>
#include "ui_ScannerWindow.h"

#include "modules/08_devicemgmt/IScannerCamera.h"   // StereoFrame（帧出口签名）
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

    void onSliderFreqChanged(int v);
    void onSliderBackgroundChanged(int v);
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
    int m_pendingWidth = 0;
    int m_pendingHeight = 0;

    uint64_t m_frameCount = 0;
    uint64_t m_prevFrameCount = 0;
    uint64_t m_currentFps = 0;

    // 帧出口（DeviceManager::startFrameStream 注册——计数+入队，最轻量）
    void pushFrameToBuffer(const Scanner::hal::StereoFrame& frame);
};
