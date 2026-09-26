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

/// 大恒 Galaxy 双目相机（装机口径参数自 config/camera.json 由装配根喂入；
/// triggerSource＝硬件触发源，缺省 Line2；pairStrictFrameId=false＝帧号校验旁路
/// （按时间对齐交付——带宽/帧率实测实验口径，缺省 true 严格配对）
std::unique_ptr<hal::IScannerCamera> createGalaxyStereoCamera(int deviceIndexLeft,
                                                              int deviceIndexRight,
                                                              bool rotateRight180,
                                                              const std::string& triggerSource = "Line2",
                                                              bool pairStrictFrameId = true);

} // namespace Scanner::device
