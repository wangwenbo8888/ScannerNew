#pragma once
// ============================================================================
// UndoRedoStack.h — 05 双账本软删除操作账本（设计定案 D4/D5；实施计划 P1）
//
// 职责：删除操作的 undo/redo 记账——账本作用经 LedgerFn 回调发出，本栈不触
// 任何具体账本（融合累积器/obs 剔除标记的实现归 07/09，P2 落地）。
// 语义（D4/D5）：
//   - 操作粒度＝一次删除（含两组点集下标）；删除＝软删除（账本标记态）；
//   - push：入账并立即作用（removed=true）；新动作清空 redo；
//   - undo：弹栈顶→反向恢复（removed=false）→压入重做栈；
//   - redo：弹重做栈顶→再删（removed=true）→压回撤销栈；
//   - 栈深上限（默认 20，可配）：超深逐出最旧——该批固化（不可再 undo）；
//   - 会话内存态不持久化。
// 统计（D6 合账口径）：删次数/删点数/undo/redo 次数/逐出批数。
// 线程模型：编辑会话 UI 单线程属主（无锁）。
// ============================================================================
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include <opencv2/core.hpp>

#include "base/types.h"

namespace Scanner::edit {

// 一批软删除（P4 扩展：账本作用以 globalId 稳定身份为准——融合云下标会因
// 前序删除而前移；恢复（undo）需点位数据经融合回插）
struct DeleteBatch {
    std::vector<uint32_t> markerIdx;      // 选择时刻的融合云下标（留档；作用时经映射换算）
    std::vector<uint32_t> laserIdx;       // 激光侧（P4 暂空——激光云编辑后续批次）
    std::vector<int32_t> markerGlobalIds; // obs 剔除/账本映射的稳定身份
    std::vector<cv::Point3f> markerPts;   // 删除时点位（undo 恢复回插）
    std::vector<cv::Vec3f> markerNormals; // 删除时法线（同上）

    size_t pointCount() const { return markerIdx.size() + laserIdx.size(); }
    bool empty() const { return markerIdx.empty() && laserIdx.empty(); }
};

class UndoRedoStack {
public:
    // 双账本作用出口：removed=true 删除作用 / false 反向恢复（由 EditWorkflow
    // 装配期注入——路由到融合累积器移除＋obs 剔除标记，P2/P4 接线）
    using LedgerFn = std::function<void(const DeleteBatch&, bool removed)>;

    explicit UndoRedoStack(size_t maxDepth = 20);

    void setLedger(LedgerFn fn) { ledger_ = std::move(fn); }

    // 空批拒绝；入账＋立即作用（删）；清空 redo；超深逐出最旧（固化）
    Scanner::Result push(const DeleteBatch& batch);
    Scanner::Result undo();                        // 撤销栈空=失败
    Scanner::Result redo();                        // 重做栈空=失败

    bool canUndo() const { return !undoStack_.empty(); }
    bool canRedo() const { return !redoStack_.empty(); }
    size_t depth() const { return undoStack_.size(); }
    size_t maxDepth() const { return maxDepth_; }

    // D6 编辑统计（合账存档用）
    struct Stats {
        size_t deleteOps   = 0;   // 删除次数（push 计数，redo 不重复计）
        size_t deletedPts  = 0;   // 删除点数累计（同上）
        size_t undoOps     = 0;
        size_t redoOps     = 0;
        size_t evicted     = 0;   // 栈深逐出批数（已固化）
    };
    const Stats& stats() const { return stats_; }

    // 会话结束清栈（账本的应用/放弃语义归 EditWorkflow ⑤——本栈只清记账）
    void clear();

private:
    size_t maxDepth_;
    LedgerFn ledger_;
    std::vector<DeleteBatch> undoStack_;
    std::vector<DeleteBatch> redoStack_;
    Stats stats_;
};

} // namespace Scanner::edit
