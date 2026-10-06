// ============================================================================
// UndoRedoStack.cpp — 双账本软删除操作账本实现（契约见 UndoRedoStack.h）
// ============================================================================
#include "UndoRedoStack.h"

namespace Scanner::edit {

UndoRedoStack::UndoRedoStack(size_t maxDepth) : maxDepth_(maxDepth ? maxDepth : 1) {}

Scanner::Result UndoRedoStack::push(const DeleteBatch& batch) {
    if (batch.empty()) return Scanner::Result::fail("UndoRedoStack: 空批不入账");
    if (ledger_) ledger_(batch, true);                 // 立即作用（软删除）
    undoStack_.push_back(batch);
    if (undoStack_.size() > maxDepth_) {               // 逐出最旧——该批固化
        undoStack_.erase(undoStack_.begin());
        ++stats_.evicted;
    }
    redoStack_.clear();                                // 新动作清 redo（D5）
    ++stats_.deleteOps;
    stats_.deletedPts += batch.pointCount();
    return Scanner::Result::ok();
}

Scanner::Result UndoRedoStack::undo() {
    if (undoStack_.empty()) return Scanner::Result::fail("UndoRedoStack: 撤销栈空");
    DeleteBatch b = std::move(undoStack_.back());
    undoStack_.pop_back();
    if (ledger_) ledger_(b, false);                    // 反向恢复
    redoStack_.push_back(std::move(b));
    ++stats_.undoOps;
    return Scanner::Result::ok();
}

Scanner::Result UndoRedoStack::redo() {
    if (redoStack_.empty()) return Scanner::Result::fail("UndoRedoStack: 重做栈空");
    DeleteBatch b = std::move(redoStack_.back());
    redoStack_.pop_back();
    if (ledger_) ledger_(b, true);                     // 再删
    undoStack_.push_back(std::move(b));
    ++stats_.redoOps;
    return Scanner::Result::ok();
}

void UndoRedoStack::clear() {
    undoStack_.clear();
    redoStack_.clear();
}

} // namespace Scanner::edit
