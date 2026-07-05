#pragma once
namespace Scanner::sdk {
// 接入端 B（二次开发 API 门面）。G1：实现在 sdk/，组合各层能力。
class IScannerSDK { public: virtual ~IScannerSDK() = 0; };
inline IScannerSDK::~IScannerSDK() = default;
}
