#pragma once
// ============================================================================
// IAuth.h — 认证接口（Crosscut 层）
// ============================================================================

#include "common/types.h"
#include <string>

namespace Scanner::crosscut {

class IAuth {
public:
    virtual ~IAuth() = default;

    /// 登录
    virtual Result login(const std::string& username, const std::string& password) = 0;

    /// 登出
    virtual Result logout() = 0;

    /// 是否已认证
    virtual bool isAuthenticated() const = 0;

    /// 获取当前用户
    virtual std::string getCurrentUser() const = 0;

    /// 检查权限
    virtual bool hasPermission(const std::string& permission) const = 0;
};

} // namespace Scanner::crosscut
