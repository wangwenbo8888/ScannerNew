#pragma once
// ============================================================================
// jmw_pm_contract.h — PipelineManager 发布契约（D1-lite，261031）
//
// 与 OperatorLib 的 api/jmw_op_contract.h 同构：守护「编译 07/09 DLL 直接
// 拷贝替换即更新算法」的替换流。消费方（app/exe 编 02 等工作流）启动时
// 比对本 DLL 实时指纹 vs 编译期期望。
// ============================================================================
#include <cstdint>

extern "C" {
/// ABI 代际：契约不兼容的结构性变更时人工 +1（当前 1）
uint32_t jmw_pm_abi_version(void);
/// 契约指纹（64 位十六进制串，静态存储于本 DLL）
const char* jmw_pm_contract_hash(void);
}
