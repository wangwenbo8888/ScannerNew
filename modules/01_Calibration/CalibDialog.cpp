#include "CalibDialog.h"

CalibDialog::CalibDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(QStringLiteral("校准设备"));
    setFixedSize(300, 200);
    setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(20, 20, 20, 20);
    mainLayout->setSpacing(16);

    QLabel *titleLabel = new QLabel(QStringLiteral("选择校准类型"));
    titleLabel->setAlignment(Qt::AlignCenter);
    QFont titleFont = titleLabel->font();
    titleFont.setPointSize(14);
    titleFont.setBold(true);
    titleLabel->setFont(titleFont);
    mainLayout->addWidget(titleLabel);

    m_btnCameraCalib = new QPushButton(QStringLiteral("相机标定"));
    m_btnCameraCalib->setFixedHeight(48);
    m_btnCameraCalib->setStyleSheet(
        "QPushButton { background-color: #8B1A2B; color: white; border: none; border-radius: 6px; font-size: 14px; }"
        "QPushButton:hover { background-color: #A52A3A; }"
        "QPushButton:pressed { background-color: #6B1020; }"
    );
    mainLayout->addWidget(m_btnCameraCalib);

    m_btnLaserCalib = new QPushButton(QStringLiteral("激光器标定"));
    m_btnLaserCalib->setFixedHeight(48);
    m_btnLaserCalib->setStyleSheet(
        "QPushButton { background-color: #8B1A2B; color: white; border: none; border-radius: 6px; font-size: 14px; }"
        "QPushButton:hover { background-color: #A52A3A; }"
        "QPushButton:pressed { background-color: #6B1020; }"
    );
    mainLayout->addWidget(m_btnLaserCalib);

    connect(m_btnCameraCalib, &QPushButton::clicked, this, [this]() {
        emit cameraCalibClicked();
        accept();
    });
    connect(m_btnLaserCalib, &QPushButton::clicked, this, [this]() {
        emit laserCalibClicked();
        accept();
    });
}

CalibDialog::~CalibDialog()
{
}
