// ============================================================================
// CameraFactory.cpp — 默认双目相机工厂实现（CameraControl 细节收在本文件）
// ============================================================================
#include "CameraFactory.h"

#include "CameraControl.h"

namespace Scanner::device {

std::unique_ptr<hal::IScannerCamera> createGalaxyStereoCamera(int deviceIndexLeft,
                                                              int deviceIndexRight,
                                                              bool rotateRight180,
                                                              const std::string& triggerSource) {
    StereoPairConfig cfg;
    cfg.deviceIndexLeft = deviceIndexLeft;
    cfg.deviceIndexRight = deviceIndexRight;
    cfg.rotateRight180 = rotateRight180;
    cfg.triggerSource = triggerSource;
    return std::make_unique<CameraControl>(cfg);
}

} // namespace Scanner::device
