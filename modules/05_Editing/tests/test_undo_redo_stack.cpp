// ============================================================================
// test_undo_redo_stack.cpp — 双账本软删除账本单测（实施计划 P1）
// ============================================================================
#include <gtest/gtest.h>

#include "UndoRedoStack.h"

#include <vector>

using namespace Scanner::edit;

namespace {
DeleteBatch mk(std::vector<uint32_t> m, std::vector<uint32_t> l = {}) {
    return DeleteBatch{std::move(m), std::move(l)};
}
} // namespace

// 记录账本作用的轨迹（batch 标记点数+removed），供断言
namespace {
struct Trace {
    std::vector<std::pair<size_t, bool>> calls;       // (点数, removed)
    void record(const DeleteBatch& b, bool removed) { calls.emplace_back(b.pointCount(), removed); }
};
}

// —— push 立即作用＋undo/redo 往返 ——
TEST(UndoRedo, PushAppliesAndUndoRestores) {
    Trace t;
    UndoRedoStack s;
    s.setLedger([&t](const DeleteBatch& b, bool r) { t.record(b, r); });
    ASSERT_TRUE(s.push(mk({1, 2, 3})).success);
    ASSERT_TRUE(s.push(mk({}, {7})).success);
    ASSERT_TRUE(s.undo().success);                     // 弹 {7} → 恢复
    ASSERT_TRUE(s.redo().success);                     // {7} → 再删
    ASSERT_TRUE(s.undo().success);
    ASSERT_TRUE(s.undo().success);                     // 弹 {1,2,3} → 恢复
    EXPECT_FALSE(s.canUndo());
    // 轨迹：删(3) 删(1) 恢复(1) 删(1) 恢复(1) 恢复(3)
    std::vector<std::pair<size_t, bool>> expect{{3,true},{1,true},{1,false},{1,true},{1,false},{3,false}};
    ASSERT_EQ(t.calls.size(), expect.size());
    for (size_t i = 0; i < expect.size(); ++i) {
        EXPECT_EQ(t.calls[i].first, expect[i].first);
        EXPECT_EQ(t.calls[i].second, expect[i].second) << "at " << i;
    }
}

// —— 新动作清 redo ——
TEST(UndoRedo, NewPushClearsRedo) {
    UndoRedoStack s;
    ASSERT_TRUE(s.push(mk({1})).success);
    ASSERT_TRUE(s.undo().success);
    EXPECT_TRUE(s.canRedo());
    ASSERT_TRUE(s.push(mk({2})).success);             // 新动作
    EXPECT_FALSE(s.canRedo());                        // redo 已清
    EXPECT_FALSE(s.redo().success);
}

// —— 栈深上限逐出最旧（固化） ——
TEST(UndoRedo, DepthCapEvictsOldest) {
    Trace t;
    UndoRedoStack s(3);
    s.setLedger([&t](const DeleteBatch& b, bool r) { t.record(b, r); });
    for (uint32_t i = 0; i < 5; ++i) ASSERT_TRUE(s.push(mk({i})).success);
    EXPECT_EQ(s.depth(), 3u);                         // 5 入 3 深 → 逐出 2
    EXPECT_EQ(s.stats().evicted, 2u);
    // 只能 undo 到第 3 批（下标 2）
    ASSERT_TRUE(s.undo().success);
    EXPECT_EQ(s.stats().undoOps, 1u);
    // 撤到只剩 2 批；逐出的两批（0/1）不可达
    EXPECT_EQ(s.depth(), 2u);
}

// —— 统计口径（D6）：redo 不重复计删 ——
TEST(UndoRedo, StatsCounters) {
    UndoRedoStack s;
    ASSERT_TRUE(s.push(mk({1, 2})).success);          // 删 1 次 2 点
    ASSERT_TRUE(s.push(mk({3}, {4})).success);        // 删 1 次 2 点
    ASSERT_TRUE(s.undo().success);
    ASSERT_TRUE(s.redo().success);                    // redo 不计删
    EXPECT_EQ(s.stats().deleteOps, 2u);
    EXPECT_EQ(s.stats().deletedPts, 4u);
    EXPECT_EQ(s.stats().undoOps, 1u);
    EXPECT_EQ(s.stats().redoOps, 1u);
}

// —— 空批拒绝／空栈失败 ——
TEST(UndoRedo, InvalidOps) {
    UndoRedoStack s;
    EXPECT_FALSE(s.push(mk({})).success);
    EXPECT_FALSE(s.undo().success);
    EXPECT_FALSE(s.redo().success);
}

// —— clear 清栈不清统计语义（会话结束回收） ——
TEST(UndoRedo, Clear) {
    UndoRedoStack s;
    ASSERT_TRUE(s.push(mk({1})).success);
    s.clear();
    EXPECT_FALSE(s.canUndo());
    EXPECT_FALSE(s.canRedo());
}
