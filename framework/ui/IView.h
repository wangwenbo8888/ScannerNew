#pragma once
// ============================================================================
// IView.h — 视图接口（UI 层）
// ============================================================================

#include "common/types.h"

namespace Scanner::ui {

class IView {
public:
    virtual ~IView() = default;

    /// 渲染视图
    virtual Result render() = 0;

    /// 显示/隐藏
    virtual Result show() = 0;
    virtual Result hide() = 0;

    /// 是否可见
    virtual bool isVisible() const = 0;
};

} // namespace Scanner::ui
