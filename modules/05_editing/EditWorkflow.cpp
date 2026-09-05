// ============================================================================
// EditWorkflow.cpp — 编辑会话主体实现（契约见 EditWorkflow.h；编入 scan_demo）
// ============================================================================
#include "EditWorkflow.h"

#include <chrono>

#include "pipelines/scan/FuseConsumer.h"        // IMarkerFuse/CloudViewHandle/ISceneFeed
#include "pipelines/scan/FrameObsAccumulator.h"
#include "pipelines/scan/ScanTypes.h"           // MarkerPoint3D

#include <spdlog/spdlog.h>
#include "jmw_logging.h"

namespace Scanner::edit {

using namespace Scanner::pipeline;

Scanner::Result EditWorkflow::enter(Deps deps) {
    if (active_) return Scanner::Result::fail("编辑会话已在进行");
    if (!deps.markerFuse || !deps.obs)
        return Scanner::Result::fail("编辑会话装配失败：融合云/obs 账本不可达");
    if (deps.markerSnapshot.empty())
        return Scanner::Result::fail("编辑会话装配失败：标志点云为空");

    deps_ = std::move(deps);
    snapshot_ = deps_.markerSnapshot;
    undo_.clear();
    active_ = true;

    // 快照法线（回插质量保持）：融合云自带法线——从账本读取对位拷贝
    const auto& fused = deps_.markerFuse->fusedPoints();
    snapshotNormals_.resize(fused.size());
    for (size_t i = 0; i < fused.size(); ++i)
        snapshotNormals_[i] = cv::Vec3f(fused[i].nx, fused[i].ny, fused[i].nz);

    // 身份映射初始恒等（就绪态云序＝末次推送序——适配器 globalId=下标）
    idxToGlobalId_.resize(fused.size());
    for (size_t i = 0; i < fused.size(); ++i) idxToGlobalId_[i] = static_cast<int32_t>(i);

    // 双账本作用出口（D4）：删除=融合云移除＋obs 剔除；恢复=融合回插＋obs 复原
    undo_.setLedger([this](const DeleteBatch& b, bool removed) { applyLedger(b, removed); });

    sessionOpenMs_ = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();

    JMW_LOG_INFO("05-EditWorkflow", "[EditWorkflow] ①开账：会话={} 标志点={}（渲染同步{}）",
                 sessionOpenMs_, snapshot_.size(), deps_.sceneFeed ? "on" : "off");
    return Scanner::Result::ok("编辑会话开始");
}

UndoRedoStack::Stats EditWorkflow::exit(bool discard) {
    if (!active_) return undo_.stats();
    size_t undone = 0;
    if (discard) {                                   // 放弃：逐批恢复＋清栈（D5⑤）
        while (undo_.canUndo()) {
            if (!undo_.undo().success) break;
            ++undone;
        }
    }
    const auto st = undo_.stats();                   // D6 合账统计
    undo_.clear();
    active_ = false;
    JMW_LOG_INFO("05-EditWorkflow",
                 "[EditWorkflow] ⑤合账：{}——删{}次/{}点 undo{}次 redo{}次 逐出{}批 | "
                 "终态标志点={}（{}放弃恢复）",
                 discard ? "放弃" : "应用", st.deleteOps, st.deletedPts, st.undoOps,
                 st.redoOps, st.evicted, idxToGlobalId_.size(), undone);
    idxToGlobalId_.clear();
    snapshot_.clear();
    snapshotNormals_.clear();
    return st;
}

Scanner::Result EditWorkflow::deleteByPolygon(const std::vector<cv::Point2f>& polygon,
                                              const cv::Matx44d& view, const cv::Matx44d& proj,
                                              int vpW, int vpH,
                                              DepthMode depth, ObjectType type) {
    if (!active_) return Scanner::Result::fail("编辑会话未开启");
    selector_.setCamera(view, proj, vpW, vpH);
    SelectionInput cand;
    cand.markerPts = snapshot_;                      // 就绪态云不变——会话快照即当前
    const auto sel = selector_.select(polygon, cand, depth, type);
    if (sel.empty()) return Scanner::Result::ok("圈选未命中");

    DeleteBatch b;
    b.markerIdx = sel.markerIdx;
    b.markerGlobalIds.reserve(sel.markerIdx.size());
    b.markerPts.reserve(sel.markerIdx.size());
    b.markerNormals.reserve(sel.markerIdx.size());
    for (uint32_t idx : sel.markerIdx) {
        b.markerGlobalIds.push_back(idxToGlobalId_[idx]);
        b.markerPts.push_back(snapshot_[idx]);
        b.markerNormals.push_back(snapshotNormals_[idx]);
    }
    const auto r = undo_.push(b);                    // 入账即作用（LedgerFn）
    return r;
}

Scanner::Result EditWorkflow::undo() {
    if (!active_) return Scanner::Result::fail("编辑会话未开启");
    return undo_.undo();
}
Scanner::Result EditWorkflow::redo() {
    if (!active_) return Scanner::Result::fail("编辑会话未开启");
    return undo_.redo();
}

void EditWorkflow::applyLedger(const DeleteBatch& b, bool removed) {
    if (b.markerGlobalIds.empty()) return;
    if (removed) {
        // globalId → 当前下标（逆映射；就绪态编辑期云只经本口变动——映射准确）
        std::vector<uint32_t> cur;
        cur.reserve(b.markerGlobalIds.size());
        for (int32_t gid : b.markerGlobalIds) {
            for (uint32_t i = 0; i < idxToGlobalId_.size(); ++i)
                if (idxToGlobalId_[i] == gid) { cur.push_back(i); break; }
        }
        if (cur.empty()) return;                     // 已不可达（前序逐出固化等）
        auto st = deps_.markerFuse->removePoints(cur);
        if (!st.success) {
            JMW_LOG_WARN("05-EditWorkflow", "[EditWorkflow] 融合云移除失败: {}", st.message);
            return;
        }
        deps_.obs->excludeMarkerObs(
            std::vector<int>(b.markerGlobalIds.begin(), b.markerGlobalIds.end()), true);

        // 映射随删压缩（removePoints 幸存按原序前移——与 cur 无关的位次不变）
        std::vector<bool> drop(idxToGlobalId_.size(), false);
        for (uint32_t i : cur) drop[i] = true;
        std::vector<int32_t> kept;
        kept.reserve(idxToGlobalId_.size() - cur.size());
        for (size_t i = 0; i < idxToGlobalId_.size(); ++i)
            if (!drop[i]) kept.push_back(idxToGlobalId_[i]);
        idxToGlobalId_.swap(kept);
    } else {
        // 恢复：点位经融合回插（同位重建体素；编辑期无新帧竞争——首点即原点）
        std::vector<calib::MarkerPoint3D> pts;
        pts.reserve(b.markerPts.size());
        for (size_t i = 0; i < b.markerPts.size(); ++i) {
            calib::MarkerPoint3D p{};
            p.x = b.markerPts[i].x;  p.y = b.markerPts[i].y;  p.z = b.markerPts[i].z;
            p.nx = b.markerNormals[i][0];
            p.ny = b.markerNormals[i][1];
            p.nz = b.markerNormals[i][2];
            p.globalId = b.markerGlobalIds[i];
            pts.push_back(p);
        }
        const double I9[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        const double T0[3] = {0, 0, 0};
        deps_.markerFuse->fuse(pts, I9, T0);
        deps_.obs->excludeMarkerObs(
            std::vector<int>(b.markerGlobalIds.begin(), b.markerGlobalIds.end()), false);
        // 回插追加尾部——映射按批序追加（fuse 保序落位）
        idxToGlobalId_.insert(idxToGlobalId_.end(), b.markerGlobalIds.begin(),
                              b.markerGlobalIds.end());
    }
    pushViewSync();                                  // ④ 渲染同步
}

void EditWorkflow::pushViewSync() {
    if (!deps_.sceneFeed) return;
    CloudViewHandle h;
    h.hostMarker = &deps_.markerFuse->fusedPoints(); // 稳定存储（适配器保证）
    h.deviceLaser = nullptr;
    h.hostLaser = nullptr;
    h.laserCount = 0;
    deps_.sceneFeed->pushCloudSnapshot(h);
}

} // namespace Scanner::edit
