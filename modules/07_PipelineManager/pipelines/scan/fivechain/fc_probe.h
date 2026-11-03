#pragma once
// fc_probe.h — 五链中间结果抽检转储（design: docs/plans/2026-11-03-五链中间结果抽检转储-design.md）
// 帧号命中 config/fc_dump.json probe.frames 时，把该帧全链中间结果按 scan_replay
// 沙箱命名落盘 diagnostics/fc_probe/cycle_NNNN/。默认关＝零开销。
//
// 【v2·快照制】原方案（融合线程 E 段直读 GPU 视图）被质量审查否决：
//  - 错帧：undistort/interp/steger/分离掩膜等 GPU 产物是算子持久缓冲的 ROI 视图，
//    契约「帧内消费、有效期至下一次 Execute」；maxInflight=56 下融合线程滞后，
//    E 段读到的几乎必然是后续帧数据＝错误证据。
//  - 捕获窗禁令：1b 大图 Global 模式捕获窗内任何线程 cudaMalloc/cudaStreamCreate
//    非法（harness.h 灰环预分配段自写禁令）——probe 运行期建流/扩 GpuMat 会撞窗。
//  - 在线缺口：在线路径 prod.matchV3/prod 从未被赋值（chain2 局部量）、CCL 走
//    CPU 版不填 d_labeledMask——三类在 E 段永远拿不到。
// v2 改产出线程快照：fc_executor 在「该帧数据尚未被后续帧覆写」的时机把 GPU
// 产物 D2H 进 ProbeSnapshot（宿主 Mat/vector），E 段 maybeDump 只读宿主数据＋
// prod 宿主字段，纯 CPU 落盘（零 CUDA 调用）。画法/字段遍历照抄沙箱 harness.cpp
// dumpCycle 段（同仓库移植）；算子与公共头零改动。本头不 include harness.h
//（防 FiveChainAdapter.h ↔ harness.h 头环）——CycleProductsCompat 前置声明即可。
#ifndef JMW_PM_FIVECHAIN_FC_PROBE_H
#define JMW_PM_FIVECHAIN_FC_PROBE_H

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace scr { struct CycleProductsCompat; }   // 宿主产物真身在 harness.h

namespace jmw::probe {

/// probe 配置（config/fc_dump.json 的 probe 段；任何异常静默回默认＝关）
struct ProbeConfig {
    bool enabled = false;
    long maxFrames = 1000;          // app 侧限量停采阈值（probe 侧仅透传/记录）
    std::set<long> frames;          // 挑帧帧号集合（1 起，流水线处理序号）
    static ProbeConfig load();      // 读 config/fc_dump.json 的 probe 段
};

/// 【v2】命中帧 GPU 产物的宿主快照——产出线程（1a/1b/chain2）在该帧数据未被
/// 后续帧覆写的时机 D2H 填充；融合线程 E 段（maybeDump）只读宿主数据。
/// 仅命中帧分配（FcJob::probeSnap unique_ptr——56 槽不常驻），E 段消费后即 reset。
/// 点集类字段为「容量宽」下载（延迟算子精确计数在排水后才知），dump 侧按 prod
/// 计数字段截断；掩膜类为整幅二值 Mat。缺失段＝该产物跳过＋WARN（宁缺勿错）。
struct ProbeSnapshot {
    cv::Mat laserMaskL, laserMaskR;        ///< 激光掩膜（1a lmask 环槽内容，CV_8UC1）
    cv::Mat markingMaskL, markingMaskR;    ///< 标记点掩膜（sep 内部缓冲视图内容）
    cv::Mat binaryL, binaryR;              ///< 分离前端二值图（d_step1_binary——0-binary.png 源）
    std::vector<cv::Point2f> stegerL, stegerR;   ///< 激光中心（摊平；截断＝steger totalPointCount）
    std::vector<cv::Point2f> undistL, undistR;   ///< 矫正后激光点（与 steger 点一一对应）
    std::vector<cv::Point2f> interpL, interpR;   ///< 极线插值点（截断＝interp interpCount）
    std::vector<cv::Point2f> matchL, matchR;     ///< v3 匹配对（chain2 段宿主直拷/GPU D2H，精确 N）
    std::vector<int> matchLine;                  ///< 匹配方线号（逐对，L/R 共用）
    std::vector<cv::Point3f> laser3d;            ///< 激光 3D（chain2 E 块宿主拷出，精确 N）
    // 其余（标志点链/CCL 组件/位姿/统计）本就是 prod 宿主字段，E 段直接读
};

/// 挑帧落盘器：fc_executor 产出线程做 hit 判定与快照存储，E 段按帧消费。
/// 非命中帧仅一次 set 查找＝零开销；命中帧产出线程 D2H（几 MB 级）＋E 段写盘。
class FrameProbe {
public:
    void configure(const ProbeConfig& c) { cfg_ = c; }
    bool enabled() const { return cfg_.enabled; }
    size_t frameCount() const { return cfg_.frames.size(); }
    /// idx 0 起，帧号=idx+1；命中 probe.frames 则 true（产出线程命中判定——
    /// 非命中帧仅此一次 set 查找，零 GPU 操作零分配）
    bool hit(long idx) const {
        return cfg_.enabled && !cfg_.frames.empty()
               && cfg_.frames.find(idx + 1) != cfg_.frames.end();
    }
    /// 命中帧全套落盘：优先消费快照宿主数据，prod 宿主字段照旧直读；
    /// 快照缺段＝该产物跳过＋一次 WARN（宁缺勿错——绝不回退读过期 GPU 视图）
    void maybeDump(long idx, const scr::CycleProductsCompat& p,
                   const ProbeSnapshot& snap);
    /// probe 开时逐帧追加 rt.csv（列格式同 JMW_FC_FRAME_DUMPS 的 rt.csv；
    /// 帧列＝idx+1，与 cycle_NNNN 目录号一致）
    void appendRt(long idx, const scr::CycleProductsCompat& p);

private:
    ProbeConfig cfg_;
    bool dirEnsured_ = false;   // appendRt 目标目录只建一次（首帧）
    bool rtStarted_ = false;    // rt.csv 首帧 trunc 覆盖旧表，此后只 append
};

} // namespace jmw::probe

#endif // JMW_PM_FIVECHAIN_FC_PROBE_H
