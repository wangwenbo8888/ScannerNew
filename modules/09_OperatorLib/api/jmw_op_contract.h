#pragma once
// ============================================================================
// jmw_op_contract.h — OperatorLib 发布契约（D1-lite，261031）
//
// 用户口径：编译 OperatorLib.dll / PipelineManager.dll 后直接拷贝替换即
// 可更新算法。本头与其导出函数是该替换的安全守护：
//   · jmw_op_contract_hash() —— DLL 内嵌的「公共头指纹」（构建期 SHA-256）
//   · 消费方（07_PipelineManager）编译期持有同一指纹；启动时比对，
//     不一致＝公共头已变＝单纯换 DLL 不安全 → 拒绝并要求全量重编
//   · 只改算子实现（头不动）→ 指纹不变 → 换 DLL 直接生效
//
// 本 api/ 目录是 OperatorLib 对外发布的契约面，布局冻结（v1）。
// ============================================================================
#include <cstdint>

extern "C" {
/// ABI 代际：契约不兼容的结构性变更时人工 +1（当前 1）
uint32_t jmw_op_abi_version(void);
/// 契约指纹（64 位十六进制串，静态存储于本 DLL）
const char* jmw_op_contract_hash(void);
}
