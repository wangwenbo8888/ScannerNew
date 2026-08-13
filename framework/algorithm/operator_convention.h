#pragma once
// ============================================================================
// operator_convention.h — 算子约定（Algorithm 层）
//
// 算子是纯函数/独立类型，无基类（G5 决策）。
// 命名空间：calib::（迁移算子保留），Scanner::algorithm（框架契约）
// ============================================================================

#include "common/types.h"
#include <string>

namespace Scanner::algorithm {

// ============================================================================
// 算子契约级别
// ============================================================================
// 所有算子返回 Scanner::Result
// 算子不持有状态（无副作用），输入输出通过参数传递
// 算子可以是：
//   - 自由函数 (纯函数)
//   - 函数对象 ( functor)
//   - 静态成员函数
//   - Lambda
// 但不是继承自某个基类的多态对象

// ============================================================================
// 算子生命周期（框架调用）
// ============================================================================
// Execute: 执行算子核心逻辑
// Destroy: 释放算子持有的资源（如有）
// Warmup: 预热（如预分配 GPU 内存）

// ============================================================================
// AlgorithmRegistry — 算子注册表（可选，用于动态发现）
// ============================================================================
template<typename Signature>
class AlgorithmRegistry {
public:
    using Factory = std::function<Signature>;

    void registerOperator(const std::string& key, Factory factory) {
        factories_[key] = std::move(factory);
    }

    Factory getOperator(const std::string& key) const {
        auto it = factories_.find(key);
        if (it != factories_.end()) return it->second;
        return nullptr;
    }

    bool hasOperator(const std::string& key) const {
        return factories_.count(key) > 0;
    }

private:
    std::unordered_map<std::string, Factory> factories_;
};

} // namespace Scanner::algorithm
