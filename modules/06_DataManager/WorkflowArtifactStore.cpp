#include "WorkflowArtifactStore.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>
#include "jmw_logging.h"

namespace fs = std::filesystem;

namespace Scanner::data {

// ============================================================================
// 文件仓实现
// ============================================================================

// 版本头：魔术字(8B) + formatVersion(4B) + dataLen(4B) = 16B 固定头
static constexpr char kMagic[8] = { 'J','M','W','A','R','T','1' };
static constexpr uint32_t kHeaderSize = 16;

FileArtifactStore::FileArtifactStore(const std::string& baseDir) : baseDir_(baseDir) {
    std::error_code ec;
    fs::create_directories(baseDir_, ec);
    if (ec) {
        JMW_LOG_WARN("06-ArtifactStore", "[ArtifactStore] 基目录创建失败 {}: {}", baseDir_, ec.message());
    }
}

std::string FileArtifactStore::filePathFor(const std::string& key) const {
    // sanitize：保留字母数字/点/下划线/连字符，其余替换 '_'——跨平台安全
    std::string safe;
    safe.reserve(key.size());
    for (char c : key) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_' || c == '-')
            safe += c;
        else
            safe += '_';
    }
    if (safe.empty() || safe == "." || safe == "..") safe = "_empty";
    return (fs::path(baseDir_) / (safe + ".art")).string();
}

bool FileArtifactStore::put(const std::string& key, const std::vector<unsigned char>& data) {
    const std::string path = filePathFor(key);
    const std::string tmp = path + ".tmp";
    try {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) return false;
        // 版本头
        f.write(kMagic, 8);
        const uint32_t ver = 1;
        f.write(reinterpret_cast<const char*>(&ver), 4);
        const uint32_t len = static_cast<uint32_t>(data.size());
        f.write(reinterpret_cast<const char*>(&len), 4);
        f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        f.close();
        if (!f.good()) { fs::remove(tmp); return false; }
        // 原子 rename（同卷）
        fs::rename(tmp, path);
        return true;
    } catch (const std::exception& e) {
        JMW_LOG_WARN("06-ArtifactStore", "[ArtifactStore] put('{}') 异常: {}", key, e.what());
        return false;
    }
}

bool FileArtifactStore::get(const std::string& key, std::vector<unsigned char>& data) const {
    const std::string path = filePathFor(key);
    try {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return false;
        char magic[8];
        f.read(magic, 8);
        if (std::memcmp(magic, kMagic, 8) != 0) return false;
        uint32_t ver = 0, len = 0;
        f.read(reinterpret_cast<char*>(&ver), 4);
        f.read(reinterpret_cast<char*>(&len), 4);
        if (ver != 1) {
            JMW_LOG_WARN("06-ArtifactStore", "[ArtifactStore] get('{}') 版本不匹配 {}≠1", key, ver);
            return false;
        }
        data.resize(len);
        f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(len));
        return f.good() || len == 0;
    } catch (const std::exception& e) {
        JMW_LOG_WARN("06-ArtifactStore", "[ArtifactStore] get('{}') 异常: {}", key, e.what());
        return false;
    }
}

std::vector<std::string> FileArtifactStore::list() const {
    std::vector<std::string> keys;
    std::error_code ec;
    for (fs::directory_iterator it(baseDir_, ec), end; it != end && !ec; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.size() > 4 && name.substr(name.size() - 4) == ".art")
            keys.push_back(name.substr(0, name.size() - 4));
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

bool FileArtifactStore::remove(const std::string& key) {
    std::error_code ec;
    return fs::remove(filePathFor(key), ec);
}

// ============================================================================
// 类型化序列化（POD 平铺——版本号首字段，读方校验；布局变化=升 version）
// ============================================================================

namespace {
void appendU32(std::vector<unsigned char>& out, uint32_t v) {
    out.resize(out.size() + 4);
    std::memcpy(out.data() + out.size() - 4, &v, 4);
}
void appendU64(std::vector<unsigned char>& out, uint64_t v) {
    out.resize(out.size() + 8);
    std::memcpy(out.data() + out.size() - 8, &v, 8);
}
struct Reader {
    const unsigned char* p;
    size_t n, off = 0;
    bool ok = true;
    template <typename T> T read() {
        if (off + sizeof(T) > n) { ok = false; return T{}; }
        T v;
        std::memcpy(&v, p + off, sizeof(T));
        off += sizeof(T);
        return v;
    }
};
} // namespace

// —— 位姿数组：[ver u32][count u32][frameId u64 + R×9 f64 + T×3 f64] × N ——

bool serializePoses(const std::vector<ArtifactPose>& poses, std::vector<unsigned char>& out) {
    appendU32(out, 1);                                        // version
    appendU32(out, static_cast<uint32_t>(poses.size()));
    for (const auto& ps : poses) {
        appendU64(out, ps.frameId);
        out.resize(out.size() + 9 * 8 + 3 * 8);
        unsigned char* dst = out.data() + out.size() - (9 * 8 + 3 * 8);
        std::memcpy(dst, ps.R, 9 * 8);
        std::memcpy(dst + 9 * 8, ps.T, 3 * 8);
    }
    return true;
}

bool deserializePoses(const std::vector<unsigned char>& in, std::vector<ArtifactPose>& poses) {
    Reader r{ in.data(), in.size() };
    const uint32_t ver = r.read<uint32_t>();
    if (!r.ok || ver != 1) return false;
    const uint32_t count = r.read<uint32_t>();
    if (!r.ok) return false;
    poses.clear();
    poses.reserve(count);
    for (uint32_t i = 0; i < count && r.ok; ++i) {
        ArtifactPose ps{};
        ps.frameId = r.read<uint64_t>();
        for (int j = 0; j < 9; ++j) ps.R[j] = r.read<double>();
        for (int j = 0; j < 3; ++j) ps.T[j] = r.read<double>();
        poses.push_back(ps);
    }
    return r.ok;
}

// —— GBA 标志点：[ver u32][count u32][x y z f64×3 + globalId i32 + covis i32] × N ——

bool serializeMarkers(const std::vector<ArtifactMarker>& markers, std::vector<unsigned char>& out) {
    appendU32(out, 1);
    appendU32(out, static_cast<uint32_t>(markers.size()));
    for (const auto& m : markers) {
        out.resize(out.size() + 3 * 8 + 4 + 4);
        unsigned char* dst = out.data() + out.size() - (3 * 8 + 4 + 4);
        std::memcpy(dst, &m.x, 8);
        std::memcpy(dst + 8, &m.y, 8);
        std::memcpy(dst + 16, &m.z, 8);
        std::memcpy(dst + 24, &m.globalId, 4);
        std::memcpy(dst + 28, &m.covisCount, 4);
    }
    return true;
}

bool deserializeMarkers(const std::vector<unsigned char>& in, std::vector<ArtifactMarker>& markers) {
    Reader r{ in.data(), in.size() };
    const uint32_t ver = r.read<uint32_t>();
    if (!r.ok || ver != 1) return false;
    const uint32_t count = r.read<uint32_t>();
    if (!r.ok) return false;
    markers.clear();
    markers.reserve(count);
    for (uint32_t i = 0; i < count && r.ok; ++i) {
        ArtifactMarker m{};
        m.x = r.read<double>();
        m.y = r.read<double>();
        m.z = r.read<double>();
        m.globalId = r.read<int32_t>();
        m.covisCount = r.read<int32_t>();
        markers.push_back(m);
    }
    return r.ok;
}

// —— 会话元信息：[ver u32][frameCount u64][initialRMSE f64][finalRMSE f64]
//    [quality i32][gbaSuccess i32][laserReplayed i32][elapsedMs u64] ——

bool serializeSessionMeta(const ArtifactSessionMeta& meta, std::vector<unsigned char>& out) {
    appendU32(out, 1);
    appendU64(out, meta.frameCount);
    out.resize(out.size() + 2 * 8);
    std::memcpy(out.data() + out.size() - 16, &meta.initialRMSE, 8);
    std::memcpy(out.data() + out.size() - 8, &meta.finalRMSE, 8);
    appendU32(out, static_cast<uint32_t>(meta.quality));
    appendU32(out, static_cast<uint32_t>(meta.gbaSuccess));
    appendU32(out, static_cast<uint32_t>(meta.laserReplayed));
    appendU64(out, meta.elapsedMs);
    return true;
}

bool deserializeSessionMeta(const std::vector<unsigned char>& in, ArtifactSessionMeta& meta) {
    Reader r{ in.data(), in.size() };
    const uint32_t ver = r.read<uint32_t>();
    if (!r.ok || ver != 1) return false;
    meta.frameCount = r.read<uint64_t>();
    meta.initialRMSE = r.read<double>();
    meta.finalRMSE = r.read<double>();
    meta.quality = r.read<int32_t>();
    meta.gbaSuccess = r.read<int32_t>();
    meta.laserReplayed = r.read<int32_t>();
    meta.elapsedMs = r.read<uint64_t>();
    return r.ok;
}

} // namespace Scanner::data
