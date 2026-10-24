// scr_config.cpp — 配置装载实现（全容错，失败不抛异常）
#include "scr_config.h"

#include <filesystem>
#include <fstream>
#include <sstream>

#ifndef JMW_SOURCE_ROOT
#define JMW_SOURCE_ROOT ""
#endif
#ifndef SCR_SOURCE_DIR
#define SCR_SOURCE_DIR ""
#endif

namespace scr {
namespace fs = std::filesystem;

bool ScrPaths::allExist(std::string& missing) const {
    struct Item { const std::string* p; const char* what; };
    const Item items[] = {
        {&dataDir, "paths.dataDir"}, {&cameraCalib, "paths.cameraCalib"},
        {&cmttSidecar, "paths.cmttSidecar"}};
    for (const auto& it : items) {
        if (it.p->empty() || !fs::exists(fs::path(*it.p))) {
            missing = std::string(it.what) + " 不存在: " + *it.p;
            return false;
        }
    }
    return true;
}

std::string ScrConfig::expandPath(const std::string& raw) {
    static const std::string kToken = "{JMW_SOURCE_ROOT}";
    std::string out = raw;
    const auto pos = out.find(kToken);
    if (pos != std::string::npos)
        out.replace(pos, kToken.size(), JMW_SOURCE_ROOT);
    // 相对路径以 source root 为基（outDir 等产物路径）
    if (!out.empty() && !fs::path(out).is_absolute()) {
        const auto base = fs::path(JMW_SOURCE_ROOT);
        if (!base.empty())
            out = (base / out).lexically_normal().string();
    }
    return out;
}

bool ScrConfig::fromJson(const nlohmann::json& j, ScrConfig& out, std::string& err) {
    try {
        if (j.contains("paths")) {
            const auto& p = j.at("paths");
            if (p.contains("dataDir"))     out.paths.dataDir = expandPath(p.at("dataDir").get<std::string>());
            if (p.contains("cameraCalib")) out.paths.cameraCalib = expandPath(p.at("cameraCalib").get<std::string>());
            if (p.contains("cmttSidecar")) out.paths.cmttSidecar = expandPath(p.at("cmttSidecar").get<std::string>());
            if (p.contains("cmttSidecarEven") && p.at("cmttSidecarEven").is_string())
                out.paths.cmttSidecarEven = expandPath(p.at("cmttSidecarEven").get<std::string>());
            if (p.contains("outDir"))      out.paths.outDir = expandPath(p.at("outDir").get<std::string>());
        } else {
            err = "config 缺 paths 节";
            return false;
        }
        if (j.contains("runtime")) {
            const auto& r = j.at("runtime");
            if (r.contains("maxCycles"))           out.runtime.maxCycles = r.at("maxCycles").get<int>();
            if (r.contains("nominalTemperature"))  out.runtime.nominalTemperature = r.at("nominalTemperature").get<double>();
            if (r.contains("warmupCycles"))        out.runtime.warmupCycles = r.at("warmupCycles").get<int>();
            if (r.contains("dumpFullFrames"))      out.runtime.dumpFullFrames = r.at("dumpFullFrames").get<int>();
            if (r.contains("checkCycles"))         out.runtime.checkCycles = r.at("checkCycles").get<int>();
            if (r.contains("check"))               out.runtime.check = r.at("check");
            if (r.contains("dedupPath"))           out.runtime.dedupPath = r.at("dedupPath").get<std::string>();
            if (r.contains("extractor"))           out.runtime.extractor = r.at("extractor").get<std::string>();
            if (r.contains("cclPath"))             out.runtime.cclPath = r.at("cclPath").get<std::string>();
            if (r.contains("markerOnly"))          out.runtime.markerOnly = r.at("markerOnly").get<bool>();
            if (r.contains("fuseMinMatched"))      out.runtime.fuseMinMatched = r.at("fuseMinMatched").get<int>();
            if (r.contains("fuseMaxRmseMm"))       out.runtime.fuseMaxRmseMm = r.at("fuseMaxRmseMm").get<double>();
            if (r.contains("fuseSubsetMax"))       out.runtime.fuseSubsetMax = r.at("fuseSubsetMax").get<int>();
            if (r.contains("mintDedupRadiusMm"))   out.runtime.mintDedupRadiusMm = r.at("mintDedupRadiusMm").get<double>();
            if (r.contains("flowRmseDriftRatio"))  out.runtime.flowRmseDriftRatio = r.at("flowRmseDriftRatio").get<double>();
        }
        if (out.runtime.maxCycles < 0 || out.runtime.warmupCycles < 0 ||
            out.runtime.dumpFullFrames < 0 || out.runtime.checkCycles < 0) {
            err = "runtime 各 cycle 计数字段须 >= 0";
            return false;
        }
        if (out.runtime.dedupPath != "operator" && out.runtime.dedupPath != "cpu") {
            err = "runtime.dedupPath 须为 \"operator\" 或 \"cpu\"，实为: "
                + out.runtime.dedupPath;
            return false;
        }
        if (out.runtime.cclPath != "cpu" && out.runtime.cclPath != "gpu") {
            err = "runtime.cclPath 须为 \"cpu\" 或 \"gpu\"，实为: "
                + out.runtime.cclPath;
            return false;
        }
        if (out.runtime.extractor != "steger" && out.runtime.extractor != "flat") {
            err = "runtime.extractor 须为 \"steger\" 或 \"flat\"，实为: "
                + out.runtime.extractor;
            return false;
        }
        if (out.runtime.extractor == "flat" && out.runtime.dedupPath == "cpu") {
            err = "runtime.extractor=flat 不兼容 runtime.dedupPath=cpu（flat 链无去重步骤）";
            return false;
        }
        if (j.contains("operators"))
            out.operators = j.at("operators");
        if (j.contains("fivechain")) {
            const auto& f = j.at("fivechain");
            auto getInt = [&](const char* k, int& v) { if (f.contains(k)) v = f.at(k).get<int>(); };
            auto getI64 = [&](const char* k, long long& v) { if (f.contains(k)) v = f.at(k).get<long long>(); };
            auto getDbl = [&](const char* k, double& v) { if (f.contains(k)) v = f.at(k).get<double>(); };
            if (f.contains("executor"))  out.fivechain.executor = f.at("executor").get<std::string>();
            getInt("graySlots", out.fivechain.graySlots);
            getInt("rleSlots", out.fivechain.rleSlots);
            getInt("rleCap", out.fivechain.rleCap);
            getInt("markerLanes", out.fivechain.markerLanes);
            getInt("bCap", out.fivechain.bCap);
            getInt("cCap", out.fivechain.cCap);
            getInt("dQueueCap", out.fivechain.dQueueCap);
            getInt("dPoolSlots", out.fivechain.dPoolSlots);
            getInt("eQueueCap", out.fivechain.eQueueCap);
            getInt("ePoolBlocks", out.fivechain.ePoolBlocks);
            getInt("eMaxPoints", out.fivechain.eMaxPoints);
            getInt("fCap", out.fivechain.fCap);
            getInt("trCap", out.fivechain.trCap);
            getInt("maxInflight", out.fivechain.maxInflight);
            getInt("memCapMB", out.fivechain.memCapMB);
            getI64("voxelReserve", out.fivechain.voxelReserve);
            getDbl("voxelSizeMm", out.fivechain.voxelSizeMm);
            getDbl("voxelLoadFactor", out.fivechain.voxelLoadFactor);
            getI64("voxelGrowSlots", out.fivechain.voxelGrowSlots);
            if (f.contains("pinRoles"))   out.fivechain.pinRoles = f.at("pinRoles").get<std::string>();
            if (f.contains("priorityBoost")) out.fivechain.priorityBoost = f.at("priorityBoost").get<bool>();
            if (f.contains("splitChain1"))  out.fivechain.splitChain1 = f.at("splitChain1").get<bool>();
            getInt("lmaskSlots", out.fivechain.lmaskSlots);
            getInt("holdSkip", out.fivechain.holdSkip);
            getInt("slowChainMs", out.fivechain.slowChainMs);
            if (f.contains("failInject")) out.fivechain.failInject = f.at("failInject").get<std::string>();
            getInt("kRleCapOverride", out.fivechain.kRleCapOverride);
            getInt("rleWindow", out.fivechain.rleWindow);
            getInt("eBlockSizePct", out.fivechain.eBlockSizePct);
            getInt("regWakeP99UsMax", out.fivechain.regWakeP99UsMax);
            getInt("dbgInterval", out.fivechain.dbgInterval);
            getDbl("pushBudgetMs", out.fivechain.pushBudgetMs);
            if (out.fivechain.executor != "serial" && out.fivechain.executor != "pipe"
                && out.fivechain.executor != "fivechain") {
                err = "fivechain.executor 须为 serial|pipe|fivechain，实为: "
                    + out.fivechain.executor;
                return false;
            }
            if (out.fivechain.kRleCapOverride > 0)
                out.fivechain.rleCap = out.fivechain.kRleCapOverride;   // R1 压测注入
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("config 解析异常: ") + e.what();
        return false;
    }
}

bool ScrConfig::fromFile(const std::string& path, ScrConfig& out, std::string& err) {
    std::error_code ec;
    if (!fs::exists(fs::path(path), ec)) {
        err = "config 文件不存在: " + path;
        return false;
    }
    std::ifstream ifs(path);
    if (!ifs) {
        err = "config 文件打开失败: " + path;
        return false;
    }
    try {
        nlohmann::json j = nlohmann::json::parse(ifs);
        return fromJson(j, out, err);
    } catch (const std::exception& e) {
        err = std::string("config JSON 解析失败: ") + e.what();
        return false;
    }
}

nlohmann::json ScrConfig::opParams(const std::string& opName) const {
    // 算子 Params::fromJson 均为 contains() 保护：空对象=全默认
    if (operators.contains(opName) && operators.at(opName).is_object())
        return operators.at(opName);
    return nlohmann::json::object();
}

std::string ScrConfig::defaultConfigPath() {
    // 优先级：exe 同目录 > 源码目录模板
    std::error_code ec;
    const auto exeDir = fs::current_path(ec);
    if (!ec) {
        const auto beside = exeDir / "config.json";
        if (fs::exists(beside, ec)) return beside.string();
    }
    return (fs::path(SCR_SOURCE_DIR) / "config.json").string();
}

} // namespace scr
