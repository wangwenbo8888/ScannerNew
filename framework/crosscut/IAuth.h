#pragma once
namespace Scanner::crosscut {
// 用户权限/鉴权（G1：框架结构元素，实现在 fw_crosscut）
class IAuth { public: virtual ~IAuth() = default; };
class ILogger { public: virtual ~ILogger() = default; };
class IPerfMonitor { public: virtual ~IPerfMonitor() = default; };
class ICrashHandler { public: virtual ~ICrashHandler() = default; };
class IConfig { public: virtual ~IConfig() = default; };
}
