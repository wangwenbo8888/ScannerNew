#pragma once
// ============================================================================
// frame_source.h — 帧枚举＋cycle 配对＋帧型（parity）判定
// ============================================================================
// 数据形态（实测）：20260831_171343/{left,right}/NNNNNN.png 各 201 张，
// 奇偶大小交替＝激光帧/补光帧两类。cycle＝1 激光帧＋1 补光帧（相邻 PNG 对），
// 尾帧孤儿丢弃。parity：auto＝列亮度分布判定（激光帧含横贯亮带）；置信不足
// 报错要求显式 odd|even（设计 §3.2，不猜）。
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include <cuda_runtime.h>   // raw 锁页注册（【150FPS】DMA 上传）

#include "scr_config.h"

namespace scr {

struct Cycle {
    int laserIdx = -1;    // 帧号（同帧模式：激光链与标记点链吃同一帧）
    int markerIdx = -1;   // ＝laserIdx（保留双字段以兼容既有产物命名）
    bool tubeOdd = true;  // 奇偶＝激光管号（true=奇号管；不同管需调用不同标定参数）
};

struct FrameSource {
    std::string leftDir, rightDir;
    std::vector<int> frameIds;   // 升序 PNG 序号
    std::vector<Cycle> cycles;   // 同帧模式：逐帧各成一个 cycle

    bool readFrame(int idx, bool left, cv::Mat& out, std::string& err) const;

    // —— 【150FPS·方案 B】raw 预解压模式：dataDir 含 meta.json＋left.bin/right.bin
    //    （无头原始灰度帧序列，PNG 逐字节无损等价）时启用。两 bin 一次性整载
    //    入内存（1688×1400×N ≈ 0.94GB/眼），readFrame＝零拷贝包装，消每帧
    //    ~50ms PNG 解码（产线相机直出亦无解码，此为回放口径对齐）。
    bool rawMode = false;
    int rawW = 0, rawH = 0;
    std::vector<unsigned char> rawBuf[2];   // [0]=L [1]=R 整载缓冲
    bool rawPinned[2] = {false, false};     // 【150FPS】cudaHostRegister 状态
    // 【利用率专项】逐块登记已注册区间（整块成功＝1 块；分块降级＝多块）——
    // 析构逐块注销（失败侧回滚后为空）
    std::vector<std::pair<unsigned char*, size_t>> rawReg[2];

    ~FrameSource() {   // 锁页内存须先注销再释放
        for (int side = 0; side < 2; ++side)
            for (auto& rg : rawReg[side])
                if (rg.first) cudaHostUnregister(rg.first);
    }
};

/// 【全链速度优化 2026-10-16】帧对预取器：IO 线程解码下一 cycle 的左右目 PNG，
/// 与当前帧流水线并行（PNG 解码＝回放 IO，非流水线计算——产线相机直出无此环节）。
/// 深度 1 双缓冲：pipeline 消费 i 时后台解码 i+1；IO 线程仅做 imread，无计算。
class FramePrefetcher {
public:
    FramePrefetcher(const FrameSource& src);
    ~FramePrefetcher();
    FramePrefetcher(const FramePrefetcher&) = delete;
    FramePrefetcher& operator=(const FramePrefetcher&) = delete;
    /// 取第 idx 帧对（命中预取立即返回；未命中同步解码）
    void get(int idx, cv::Mat& grayL, cv::Mat& grayR, std::string& err);
    /// 预取完成后下一帧对是否已就绪（诊断用）
    bool ready() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

/// 枚举帧＋配 cycle＋（auto 时）parity 判定；失败返 nullptr 填 err
std::unique_ptr<FrameSource> makeFrameSource(const ScrConfig& cfg, std::string& err);

/// 【150FPS·方案 B】PNG 目录 → raw 目录一次性转换：读 <src>/left|right/NNNNNN.png
/// 全量解码，写 <dst>/left.bin＋right.bin（无头逐帧连续）＋meta.json（宽高＋帧号表）。
/// 帧字节与 imread 逐位一致（PNG 无损），金样对拍不受影响。
bool convertToRaw(const std::string& srcDir, const std::string& dstDir, std::string& err);

/// 帧型判定（单帧·绝对特征）：true=疑似横贯激光亮带。同帧模式下不再用于帧型
/// 区分（所有帧均含激光＋标记点）；保留给掩膜形态单测。
bool looksLikeLaserFrame(const cv::Mat& gray);

/// 高亮像素计数（>thresh，2 步采样）——两管亮度差异的信息性统计
long countBrightPixels(const cv::Mat& gray, uchar thresh = 220);

} // namespace scr
