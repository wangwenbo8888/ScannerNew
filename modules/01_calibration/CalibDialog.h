#pragma once

#include <QDialog>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
class CalibDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CalibDialog(QWidget *parent = nullptr);
    ~CalibDialog();

signals:
    void cameraCalibClicked();
    void laserCalibClicked();

private:
    QPushButton *m_btnCameraCalib;
    QPushButton *m_btnLaserCalib;
};
