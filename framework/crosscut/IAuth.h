#pragma once
namespace Scanner::crosscut {
// 用户权限/鉴权（G1：框架结构元素，实现在 fw_crosscut）
class IAuth { public: virtual ~IAuth() = 0; };
inline IAuth::~IAuth() = default;
class ILogger { public: virtual ~ILogger() = 0; };
inline ILogger::~ILogger() = default;
class IPerfMonitor { public: virtual ~IPerfMonitor() = 0; };
inline IPerfMonitor::~IPerfMonitor() = default;
class ICrashHandler { public: virtual ~ICrashHandler() = 0; };
inline ICrashHandler::~ICrashHandler() = default;
class IConfig { public: virtual ~IConfig() = 0; };
inline IConfig::~IConfig() = default;
}
