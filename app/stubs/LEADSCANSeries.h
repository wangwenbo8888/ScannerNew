#pragma once

#include <QMainWindow>
#include <QStringList>
#include <QThreadPool>
#include <opencv2/core.hpp>

class CameraControl;

class LEADSCANSeries : public QMainWindow
{
public:
    explicit LEADSCANSeries(QWidget *parent = nullptr) : QMainWindow(parent) {}
    ~LEADSCANSeries() override = default;

    QStringList getPortNameList() { return {}; }
    CameraControl* getCameraControl() { return nullptr; }
    QThreadPool* GetThreadPool() { return nullptr; }
};
