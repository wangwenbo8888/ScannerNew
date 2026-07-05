#pragma once
namespace Scanner::service {
class IService { public: virtual ~IService() = default; virtual const char* name() const = 0; };
class StateMachine {};   // 待机/标定/扫描/后处理/故障 转移
class FaultHandler {};   // ADR 7.9 HardwareFault/AlgorithmFault → FaultOccurred
}
