#pragma once
// ============================================================================
// ScannerWindow.h — 扫描仪控制窗口（基于 LEADSCANSeries.ui）
//
// 使用 Scanner::device::CameraControl 管理相机，串口通信控制下位机。
// ============================================================================

#include <QMainWindow>
#include <QTimer>
#include <QSerialPort>
#include <QSerialPortInfo>
#include "ui_ScannerWindow.h"

#include "modules/08_devicemgmt/CameraControl.h"
#include "data/FrameBuffer.h"

class AppContext;

class ScannerWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit ScannerWindow(AppContext* appCtx = nullptr, QWidget *parent = nullptr);
    ~ScannerWindow();

    Scanner::device::CameraControl* getCameraControl();
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
    Scanner::device::CameraControl* m_cam = nullptr;
    Scanner::data::FrameBuffer* m_frameBuffer = nullptr;
    QTimer* m_fpsTimer = nullptr;
    QTimer* m_consumerTimer = nullptr;
    QComboBox* m_resCombo = nullptr;
    int m_pendingWidth = 0;
    int m_pendingHeight = 0;

    uint64_t m_frameCount = 0;
    uint64_t m_prevFrameCount = 0;
    uint64_t m_currentFps = 0;

    // 串口
    QSerialPort* m_serialPort1 = nullptr;
    QSerialPort* m_serialPort2 = nullptr;
    QStringList m_portNameList;

    void openPorts();
    void sendData(QSerialPort* port, const QString& data);
    QString buildStartCommand();
    QString buildStopCommand();
};
