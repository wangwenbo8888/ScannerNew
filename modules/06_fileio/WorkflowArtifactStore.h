#pragma once
// ============================================================================
// WorkflowArtifactStore.h — L4 产物仓库预留（框架整体 §1 DataStore／
// 扫描流水线 ADR 7.10）——本次不实现，接线批落地。
// ============================================================================

#include <string>
#include <vector>

namespace Scanner::data {

class WorkflowArtifactStore {
public:
    virtual ~WorkflowArtifactStore() = default;

    virtual bool put(const std::string& key, const std::vector<unsigned char>& data) = 0;
    virtual bool get(const std::string& key, std::vector<unsigned char>& data) const = 0;
    virtual std::vector<std::string> list() const = 0;
};

} // namespace Scanner::data
