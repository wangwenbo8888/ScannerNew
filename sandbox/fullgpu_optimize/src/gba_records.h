#pragma once
// ============================================================================
// gba_records.h — 全局优化（GBA）收尾阶段的留档记录【2026-11-16 方案定稿】
// ============================================================================
// 设计依据：归档/记录文档/方案分析-全局优化接入-2026-11-16.md §三·终。
//  · GbaObsRecord  ＝ 回放期间 runRegistration 尾快照 prevState_（本机系标记点
//    观测＋globalId；含 gid=-1 候选点——装配期过滤）。仅 regSource∈{first,flow,
//    fuse} 帧入档（hold/skip 的 prevState_ 停在旧帧＝重复观测，严禁入档）。
//  · GbaLaserRecord＝逐帧相机系激光点（live 融合前的原始点）＋live 所用位姿。
//    五链＝fuseLoop E 块拷出（槽移交前）；串行＝accumulateGlobalLaser 内拷出。
//  · 写者纪律：gbaObs_ 单写者＝配准线程（runRegistration 内）；gbaLaser_ 单写者
//    ＝融合线程（五链 fuseLoop）或主线程（串行）——无锁；读者＝收尾 GBA 阶段，
//    happens-before 由 shutdownExecutors()（fc_.reset() 真 join）建立。
//  · --repeat 多遍：遍边界 gbaClearArchives() 清零（跨遍增长＝R5 事故同款；
//    单测钉死）。内存量级：obs ~0.2MB＋laser ~130MB/遍（309 帧×~34k 点×12B）。
#include <cstdint>
#include <vector>
#include <opencv2/core.hpp>

namespace scr {

struct GbaObsRecord {
    uint64_t frameId = 0;
    double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // 本帧全局位姿（配准输出）
    double T[3] = {0, 0, 0};
    std::vector<cv::Point3d> local;              // 本机系标记点圆心（与 gids 同序）
    std::vector<int> gids;                       // 全局编号（候选期 -1，装配期过滤）
};

struct GbaLaserRecord {
    uint64_t frameId = 0;
    double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // live 融合所用位姿（重融合缺位时的原位姿兜底）
    double T[3] = {0, 0, 0};
    std::vector<float> xyz;                      // 相机系激光点，3N 交错（x,y,z,...）
};

/// 遍边界清零（--repeat 长跑防跨遍增长；Harness::gbaResetPass() 与单测共用）
inline void gbaClearArchives(std::vector<GbaObsRecord>& obs,
                             std::vector<GbaLaserRecord>& laser) {
    // shrink_to_disk 语义：连容量一起释放（130MB 级，留着只是浪费）
    std::vector<GbaObsRecord>().swap(obs);
    std::vector<GbaLaserRecord>().swap(laser);
}

}  // namespace scr
