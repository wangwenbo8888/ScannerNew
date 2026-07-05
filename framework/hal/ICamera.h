#pragma once
namespace Scanner::hal {
class ICamera { public: virtual ~ICamera() = default; };   // 通讯链边界适配器
class IMCU { public: virtual ~IMCU() = default; };
class IPlatform { public: virtual ~IPlatform() = default; };
}
