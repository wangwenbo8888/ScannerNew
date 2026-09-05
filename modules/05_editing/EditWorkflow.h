#pragma once
// ============================================================================
// EditWorkflow.h — 05 编辑会话主体（设计定案 D1–D8；实施计划 P4）
//
// 职责（05 只做 ①—⑤ 承载，账本作用经注入件）：
//   ① 门禁进入（就绪态判定归 app——canEnterEditSession；本类 enter 装配）
//   ② 装配：注入存活件（融合云/obs/渲染推送），构造会话私有件（选择引擎＋
//      撤销栈——成员持有）
//   ③ 交互承载：圈选多边形→SelectionService→DeleteBatch→撤销栈入账
//      （LedgerFn 双账本作用：融合云移除＋obs 剔除＋渲染同步）
//   ④ 旁观显示：每次账本作用后推 sceneFeed 渲染同步
//   ⑤ 回收：exit（放弃=全部 undo＋清栈；应用=已物理落地直清栈）＋D6 合账
//      统计；退回就绪态由 app 编排（本类不触扫描会话状态）
//
// 身份约定：obs globalId（适配器推送时＝融合云下标）为稳定身份；融合云下标
// 因删除前移，会话内维护 idxToGlobalId 映射（初始恒等；删除随删压缩；恢复
// 追加尾部）。账本作用一律先换算到当前下标。
//
// 线程模型：编辑会话 UI 单线程属主（无锁）。
// 编译归属：编入 scan_demo（对齐 01/02 工作流——见 app/CMakeLists.txt）。
// ============================================================================
#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "SelectionService.h"
#include "UndoRedoStack.h"
#include "base/types.h"

namespace Scanner::pipeline {
struct IMarkerFuse;
class FrameObsAccumulator;
struct ISceneFeed;
} // namespace Scanner::pipeline

namespace Scanner::edit {

class EditWorkflow {
public:
    /// 注入存活件（05 D7：就绪态扫描会话私有＋app 存活渲染推送口）
    struct Deps {
        Scanner::pipeline::IMarkerFuse* markerFuse = nullptr;      // 融合云账本
        Scanner::pipeline::FrameObsAccumulator* obs = nullptr;     // obs 账本（GBA 输入）
        Scanner::pipeline::ISceneFeed* sceneFeed = nullptr;        // 渲染同步出口
        std::vector<cv::Point3f> markerSnapshot;                   // 进入时候选快照（=融合云）
    };

    /// ① 装配进入（调用前置：app 已判定就绪态＋云非空）；重复进入=失败
    Scanner::Result enter(Deps deps);
    /// ⑤ 回收（discard=true 放弃全部编辑=逐批 undo＋清栈；false 应用=已物理
    /// 落地仅清栈）；返回 D6 统计并落日志（会话合账）
    UndoRedoStack::Stats exit(bool discard);

    bool active() const { return active_; }

    /// ③ 圈选删除：屏幕闭合多边形＋当前相机 → 三栏选择 → 入账（LedgerFn 即
    /// 作用双账本＋渲染同步）。候选＝会话快照（就绪态云不变）。选中空=无账
    /// 不入栈（ok 返回）。
    Scanner::Result deleteByPolygon(const std::vector<cv::Point2f>& polygon,
                                    const cv::Matx44d& view, const cv::Matx44d& proj,
                                    int vpW, int vpH,
                                    DepthMode depth, ObjectType type);
    /// 撤销/重做（LedgerFn 反向/再作用＋渲染同步）
    Scanner::Result undo();
    Scanner::Result redo();
    bool canUndo() const { return undo_.canUndo(); }
    bool canRedo() const { return undo_.canRedo(); }

    /// 当前工作集统计（④ 状态栏口径：标志点数/已删点数/可撤销步数）
    size_t markerCount() const { return idxToGlobalId_.size(); }
    const UndoRedoStack::Stats& stats() const { return undo_.stats(); }

private:
    void applyLedger(const DeleteBatch& b, bool removed);   // 双账本作用＋渲染同步
    void pushViewSync();                                    // ④ 渲染同步推送

    bool active_ = false;
    Deps deps_;
    UndoRedoStack undo_{20};                                // ③ 会话私有件（D5）
    SelectionService selector_;                             // ③ 三栏引擎
    std::vector<cv::Point3f> snapshot_;                     // 进入时候选（就绪态不变）
    std::vector<cv::Vec3f> snapshotNormals_;                // 快照法线（回插用）
    std::vector<int32_t> idxToGlobalId_;                    // 当前下标→稳定身份
    int64_t sessionOpenMs_ = 0;                             // D6 开账时间戳
};

} // namespace Scanner::edit
