#pragma once
// ============================================================================
// WorkflowArtifactStore.h — L4 跨阶段产物仓库（数据管理原则.md L4／框架整体
// §1 DataStore／扫描流水线 ADR 7.10）——260920 实现落地
//
// 定位：扫描↔GBA↔后处理↔编辑、标定↔扫描共享的**跨工作流/跨会话**产物归置点。
// L1-L3 管"正在算的数据"（算子内/帧内/会话内），L4 管"算完要传家的数据"。
//
// 形态：**文件仓**（目录持久化——跨会话天然可用；字节 blob 内核＋版本头）。
// 接口保持预留版签名（put/get/list 字节），上层加类型化适配器序列化。
// 大载荷（激光融合云 GB 级）由调方自行管理路径（本仓只存 manifest/索引）。
//
// 版本头：每个 artifact 文件 = [8B 魔术字 "JMWART1"][4B formatVersion]
//        [4B dataLen][dataLen bytes payload]——读方校验版本，不匹配 fail 不崩。
// ============================================================================

#include <cstdint>
#include <string>
#include <vector>

namespace Scanner::data {

/// 字节仓抽象（L4 内核——实现方可换内存/数据库/网络后端）
class WorkflowArtifactStore {
public:
    virtual ~WorkflowArtifactStore() = default;

    /// 写入（原子：先写临时文件再 rename——崩溃安全）
    virtual bool put(const std::string& key, const std::vector<unsigned char>& data) = 0;
    /// 读出（key 不存在 → false；版本不匹配 → false）
    virtual bool get(const std::string& key, std::vector<unsigned char>& data) const = 0;
    /// 列举全部 key（不含 .tmp）
    virtual std::vector<std::string> list() const = 0;
    /// 删除（可空实现）
    virtual bool remove(const std::string& key) { (void)key; return false; }
};

/// 文件仓实现（目录持久化；key 经 sanitize 后映射文件名——子目录/特殊字符安全）
class FileArtifactStore final : public WorkflowArtifactStore {
public:
    /// baseDir 不存在自动创建（递归）
    explicit FileArtifactStore(const std::string& baseDir);
    ~FileArtifactStore() override = default;

    bool put(const std::string& key, const std::vector<unsigned char>& data) override;
    bool get(const std::string& key, std::vector<unsigned char>& data) const override;
    std::vector<std::string> list() const override;
    bool remove(const std::string& key) override;

    /// 基目录（调试/导出用）
    const std::string& baseDir() const { return baseDir_; }

private:
    std::string baseDir_;
    /// key → 安全文件路径（sanitize：'/'→'_'、':'→'-'、非法字符替换）
    std::string filePathFor(const std::string& key) const;
};

// ============================================================================
// 类型化适配器（put<T>/get<T>——每类产物一个 serialize/deserialize 对）
// 格式版本由各 serialize 自带（首字段 uint32_t version）——读方校验
// ============================================================================

/// 优化后位姿数组（终局遍产物：GBA 修正后帧位姿——离线重融合/后处理基准）
struct ArtifactPose {
    uint64_t frameId;
    double R[9];
    double T[3];
};
bool serializePoses(const std::vector<ArtifactPose>& poses, std::vector<unsigned char>& out);
bool deserializePoses(const std::vector<unsigned char>& in, std::vector<ArtifactPose>& poses);

/// GBA 优化后全局标志点（终版标志点——续扫基准/后处理锚点）
struct ArtifactMarker {
    double x, y, z;
    int32_t globalId;
    int32_t covisCount;
};
bool serializeMarkers(const std::vector<ArtifactMarker>& markers, std::vector<unsigned char>& out);
bool deserializeMarkers(const std::vector<unsigned char>& in, std::vector<ArtifactMarker>& markers);

/// 会话元信息（终局遍汇总——排障/审计）
struct ArtifactSessionMeta {
    uint64_t frameCount;
    double initialRMSE;
    double finalRMSE;
    int32_t quality;           // QualityFlag int 值
    int32_t gbaSuccess;
    int32_t laserReplayed;
    uint64_t elapsedMs;
};
bool serializeSessionMeta(const ArtifactSessionMeta& meta, std::vector<unsigned char>& out);
bool deserializeSessionMeta(const std::vector<unsigned char>& in, ArtifactSessionMeta& meta);

} // namespace Scanner::data
