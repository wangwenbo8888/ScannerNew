#pragma once
namespace Scanner::ui {
class IView { public: virtual ~IView() = default; virtual void render() = 0; };
// ADR 7.8 预留：IPointCloudReadView（UI 跨层直读 Data 的只读窄接口）待实现
class IUIController { public: virtual ~IUIController() = default; };
}
