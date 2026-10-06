#pragma once
// ============================================================================
// ICloudWarehouse.h — 点云仓库流式写口（06 定义、07 消费的窄接口）
//
// 260919 补齐 06 设计图「07/09 融合结果 —pushPointCloud→ PointCloudBuffer」的
// 正式接线：融合线程（07 FuseConsumer）直写仓库，app 层「基线锁存＋整包替换」
// UI 补丁链退役。02 以 06 PointCloudBuffer 适配实现。
//
// 语义（会话两层）：
//   beginCloudSession()              会话边界——既有内容折叠为基线前缀（跨会话
//                                    只增不减，原 UI 层基线锁存语义入仓）
//   pushSessionCloud(xyz 交错 float) 本会话融合云**累计快照**（体素去重后单调
//                                    不减——替换会话层而非追加，防重复入账）
// ============================================================================

#include <cstddef>
#include <vector>

namespace Scanner::data {

class ICloudWarehouse {
public:
    virtual ~ICloudWarehouse() = default;
    virtual void beginCloudSession() = 0;
    virtual void pushSessionCloud(const std::vector<float>& xyzInterleaved) = 0;
};

} // namespace Scanner::data
