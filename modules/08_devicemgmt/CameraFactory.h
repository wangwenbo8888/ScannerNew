#pragma once
// ============================================================================
// CameraFactory.h — 默认双目相机工厂（契约出口）
//
// 跨层封装收口（2026-09-05）：装配根（app）经此构造相机——不漏实现类
// CameraControl/StereoPairConfig 细节（此前 AppContext 直 include 实现
// 头构造配置，穿透门面挂账清账）。返回 08 契约接口 IScannerCamera。
// ============================================================================
#include <memory>

#include "IScannerCamera.h"

namespace Scanner::device {

/// 大恒 Galaxy 双目相机（左 deviceIndexLeft / 右 deviceIndexRight 设备序号；
/// rotateRight180＝右图 180° 旋转装机口径）
std::unique_ptr<hal::IScannerCamera> createGalaxyStereoCamera(int deviceIndexLeft,
                                                              int deviceIndexRight,
                                                              bool rotateRight180);

} // namespace Scanner::device
