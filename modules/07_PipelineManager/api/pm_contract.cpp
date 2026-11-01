// ============================================================================
// pm_contract.cpp — PipelineManager 契约导出实现（D1-lite，261031）
// 指纹值来自构建期生成的 jmw_pm_contract_hash.h。详见 api/jmw_pm_contract.h。
// ============================================================================
#include "jmw_pm_contract.h"

#include "jmw_pm_contract_hash.h"   // 构建期生成（cmake/contract_hash.cmake）

extern "C" uint32_t jmw_pm_abi_version(void) {
    return jmw::contract::pm::abi_version;
}

extern "C" const char* jmw_pm_contract_hash(void) {
    return jmw::contract::pm::hash;
}
