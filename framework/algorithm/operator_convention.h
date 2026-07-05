#pragma once
// 算子契约（文档化约定，非基类派生 — G5 决策）：
//   算子 = 纯函数/独立类型；提供 Execute/Destroy/Warmup；
//   返回自带 Result（含 success/qualityFlag/message，QualityFlag 见 Scanner::QualityFlag）。
//   不强制继承。具体形态在 modules/09_operatorlib 集成时按算子定。
//   对齐算子规范 v1.9 三元组（Params/Result/Operator）。
namespace Scanner::algorithm {
// ADR 7.1 AlgorithmRegistry：概念性模板注册，集成时定
template <typename T> class AlgorithmRegistry {
public:
    void registerOperator(const char* /*key*/, T* /*op*/) {}
};
}
