#pragma once
// ============================================================================
// IntegrateTestDialog.h — 集成测试对话框
//
// 集成 CameraControl 采集、参数调节、图像显示、标定操作的测试界面。
// ============================================================================

#include <QDialog>
#include <QGroupBox>
#include <QPushButton>
#include <QLabel>
#include <QSlider>
#include <QTextEdit>
#include <QCheckBox>
#include <QSpinBox>
#include <QTimer>

#include "modules/08_devicemgmt/CameraControl.h"

class IntegrateTestDialog : public QDialog
{
    Q_OBJECT

public:
    explicit IntegrateTestDialog(QWidget *parent = nullptr);
    ~IntegrateTestDialog();

    void setCameraControl(Scanner::device::CameraControl* cam);

private slots:
    void onOpenCamera();
    void onCloseCamera();
    void onStartCapture();
    void onStopCapture();
    void onFrameReady();

private:
    QWidget* createControlPanel();
    QWidget* createImagePanel();
    QWidget* createParameterPanel();
    QWidget* createCalibPanel();

    void setupUILayout();

    // Camera
    Scanner::device::CameraControl* m_cam = nullptr;

    // Control panel
    QTextEdit*  m_textEditInfo;
    QPushButton* m_btnOpen;
    QPushButton* m_btnClose;
    QPushButton* m_btnStart;
    QPushButton* m_btnStop;

    // Parameter panel
    QSlider*  m_sliderExposeTime;
    QLabel*   m_labelExposeTimeValue;
    QLabel*   m_labelScannerLeftFps;
    QLabel*   m_labelScannerRightFps;
    QSlider*  m_sliderFreq;
    QLabel*   m_labelFreqValue;
    QSlider*  m_sliderBackground;
    QLabel*   m_labelBackgroundValue;
    QSlider*  m_sliderLaserLighting;
    QLabel*   m_labelLaserLightingValue;

    // Image panel
    QLabel* m_labelLeftImage;
    QLabel* m_labelRightImage;
    QLabel* m_labelSuccessCount;

    // FPS 计数
    uint64_t m_leftFrameCount = 0;
    uint64_t m_rightFrameCount = 0;
    uint64_t m_successCount = 0;
    QTimer*  m_fpsTimer = nullptr;
};
