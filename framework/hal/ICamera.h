#pragma once
namespace Scanner::hal {
class ICamera { public: virtual ~ICamera() = 0; };   // 通讯链边界适配器
inline ICamera::~ICamera() = default;
class IMCU { public: virtual ~IMCU() = 0; };
inline IMCU::~IMCU() = default;
class IPlatform { public: virtual ~IPlatform() = 0; };
inline IPlatform::~IPlatform() = default;
}
