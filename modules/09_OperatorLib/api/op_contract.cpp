// ============================================================================
// op_contract.cpp — OperatorLib 契约导出实现（D1-lite，261031）
// 指纹值来自构建期生成的 jmw_op_contract_hash.h（同文件亦供消费方作
// 编译期期望值——两侧天然对齐）。详见 api/jmw_op_contract.h。
// ============================================================================
#include "jmw_op_contract.h"

#include "jmw_op_contract_hash.h"   // 构建期生成（cmake/contract_hash.cmake）

extern "C" uint32_t jmw_op_abi_version(void) {
    return jmw::contract::op::abi_version;
}

extern "C" const char* jmw_op_contract_hash(void) {
    return jmw::contract::op::hash;
}
