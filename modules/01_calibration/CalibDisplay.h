#pragma once

#include <osg/Group>
#include <string>
#include <QWidget>

namespace calib_display {

osg::Group* buildCalibScene(const std::string& stlPath);

// 2D 标定板控件（Qt 绘制，无 3D 渲染）
class CalibBoard2D : public QWidget {
public:
    explicit CalibBoard2D(QWidget* parent = nullptr);
protected:
    void paintEvent(QPaintEvent*) override;
};

} // namespace calib_display
