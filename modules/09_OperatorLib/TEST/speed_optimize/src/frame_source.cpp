// frame_source.cpp — 帧枚举＋cycle 配对＋parity 判定实现
#include "frame_source.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

namespace scr {
namespace fs = std::filesystem;

bool FrameSource::readFrame(int idx, bool left, cv::Mat& out, std::string& err) const {
    // —— raw 模式：零拷贝包装预载缓冲切片（帧号→槽位＝frameIds 升序二分）——
    if (rawMode) {
        const auto it = std::lower_bound(frameIds.begin(), frameIds.end(), idx);
        if (it == frameIds.end() || *it != idx) {
            err = "raw 帧号越界: " + std::to_string(idx);
            return false;
        }
        const size_t slot = static_cast<size_t>(it - frameIds.begin());
        const size_t frameBytes = static_cast<size_t>(rawW) * rawH;
        unsigned char* base = const_cast<unsigned char*>(rawBuf[left ? 0 : 1].data())
                              + slot * frameBytes;
        out = cv::Mat(rawH, rawW, CV_8UC1, base);   // 外部数据零拷贝（只读消费）
        return true;
    }
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%06d", idx);
    const fs::path p = fs::path(left ? leftDir : rightDir) / (std::string(buf) + ".png");
    std::error_code ec;
    if (!fs::exists(p, ec)) {
        err = "帧不存在: " + p.string();
        return false;
    }
    out = cv::imread(p.string(), cv::IMREAD_GRAYSCALE);
    if (out.empty()) {
        err = "帧读取失败(非灰度/损坏): " + p.string();
        return false;
    }
    return true;
}

// ============================================================================
// FramePrefetcher —— 双 IO 线程（L/R 并行）＋深度 2 帧对预取
// （PNG 解码＝回放 IO，非流水线计算——产线相机直出无此环节；双线程把单帧对
// 解码墙钟减半，深度 2 覆盖处理耗时抖动）
// ============================================================================
struct FramePrefetcher::Impl {
    const FrameSource* src = nullptr;
    static constexpr int kDepth = 2;          // 预取帧对数
    struct Slot {
        cv::Mat side[2];                      // [0]=L [1]=R
        bool done[2] = {false, false};
        bool claimed[2] = {false, false};     // 该侧在产认领
        int idx = -1;
    };
    Slot slots_[kDepth];
    std::thread ioTh_[2];                     // side 0/1 各一
    std::mutex mtx;
    std::condition_variable cvWake;
    bool shutdown_ = false;

    ~Impl() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            shutdown_ = true;
        }
        cvWake.notify_all();
        for (auto& t : ioTh_) if (t.joinable()) t.join();
    }

    void ioLoop(int side) {
        for (;;) {
            int myIdx = -1;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cvWake.wait(lk, [&] {
                    if (shutdown_) return true;
                    for (int d = 0; d < kDepth; ++d)
                        if (slots_[d].idx >= 0 && !slots_[d].done[side]
                            && !slots_[d].claimed[side])
                            return true;
                    return false;
                });
                if (shutdown_) return;
                for (int d = 0; d < kDepth; ++d) {
                    if (slots_[d].idx >= 0 && !slots_[d].done[side]
                        && !slots_[d].claimed[side]) {
                        slots_[d].claimed[side] = true;
                        myIdx = slots_[d].idx;
                        break;
                    }
                }
                if (myIdx < 0) continue;
            }
            cv::Mat img;
            std::string err;
            src->readFrame(myIdx, side == 0, img, err);   // 错误在 get() 同步兜底
            std::lock_guard<std::mutex> lk(mtx);
            for (int d = 0; d < kDepth; ++d) {
                if (slots_[d].idx == myIdx) {
                    slots_[d].side[side] = std::move(img);
                    slots_[d].done[side] = true;
                    break;
                }
            }
        }
    }

    int nextFrameId(int idx) const {
        for (int id : src->frameIds)
            if (id > idx) return id;
        return -1;
    }
};

FramePrefetcher::FramePrefetcher(const FrameSource& src) : pImpl_(new Impl) {
    pImpl_->src = &src;
    for (int s = 0; s < 2; ++s)
        pImpl_->ioTh_[s] = std::thread([this, s] { pImpl_->ioLoop(s); });
}

FramePrefetcher::~FramePrefetcher() = default;

void FramePrefetcher::get(int idx, cv::Mat& grayL, cv::Mat& grayR, std::string& err) {
    bool hit = false;
    {
        std::lock_guard<std::mutex> lk(pImpl_->mtx);
        for (int d = 0; d < pImpl_->kDepth; ++d) {
            auto& s = pImpl_->slots_[d];
            if (s.idx == idx && s.done[0] && s.done[1]) {
                grayL = s.side[0];
                grayR = s.side[1];
                hit = true;
                s.idx = -1;                   // 清槽（侧数据随取随弃）
                break;
            }
        }
    }
    if (!hit) {
        if (!pImpl_->src->readFrame(idx, true, grayL, err)
            || !pImpl_->src->readFrame(idx, false, grayR, err)) {
            // err 已填
        }
    }
    // 布置新预取窗口：idx+1、idx+2（槽空闲则占位并唤醒两侧线程）
    {
        std::lock_guard<std::mutex> lk(pImpl_->mtx);
        int want = idx;
        for (int d = 0; d < pImpl_->kDepth; ++d) {
            want = pImpl_->nextFrameId(want);
            if (want < 0) break;
            auto& s = pImpl_->slots_[d];
            if (s.idx == want) continue;      // 已在产
            if (s.idx >= 0 && !(s.done[0] && s.done[1])) continue;   // 槽忙
            s.idx = want;
            s.done[0] = s.done[1] = false;
            s.claimed[0] = s.claimed[1] = false;
        }
    }
    pImpl_->cvWake.notify_all();
}

bool FramePrefetcher::ready() const {
    std::lock_guard<std::mutex> lk(pImpl_->mtx);
    for (int d = 0; d < pImpl_->kDepth; ++d)
        if (pImpl_->slots_[d].idx >= 0 && pImpl_->slots_[d].done[0]
            && pImpl_->slots_[d].done[1])
            return true;
    return false;
}

bool looksLikeLaserFrame(const cv::Mat& gray) {
    if (gray.empty() || gray.type() != CV_8UC1) return false;
    // 降采样到宽 512 再统计（判定用，省时）
    cv::Mat small = gray;
    if (gray.cols > 512)
        cv::resize(gray, small, cv::Size(512, std::max(1, gray.rows * 512 / gray.cols)));
    cv::Mat bw;
    cv::threshold(small, bw, 80, 255, cv::THRESH_BINARY);
    // 列和：亮像素多的列＝激光线竖直投影所在列
    cv::Mat colSum;
    cv::reduce(bw, colSum, 0, cv::REDUCE_SUM, CV_32S);   // 1×cols
    // 求最长连续亮列簇（colSum > rows×2：该列至少有 2 行以上亮像素，滤孤立点）
    const int n = colSum.cols;
    const int thresh = small.rows * 2;
    int best = 0, run = 0;
    for (int c = 0; c < n; ++c) {
        const int v = colSum.at<int>(0, c);
        if (v > thresh) {
            ++run;
            best = std::max(best, run);
        } else {
            run = 0;
        }
    }
    // 激光条带投影应横贯画幅相当比例（≥50% 列）
    return best >= n / 2;
}

long countBrightPixels(const cv::Mat& gray, uchar thresh) {
    if (gray.empty() || gray.type() != CV_8UC1) return -1;
    long cnt = 0;
    for (int y = 0; y < gray.rows; y += 2)
        for (int x = 0; x < gray.cols; x += 2)
            if (gray.at<uchar>(y, x) > thresh) ++cnt;
    return cnt;
}

std::unique_ptr<FrameSource> makeFrameSource(const ScrConfig& cfg, std::string& err) {
    auto src = std::make_unique<FrameSource>();
    src->leftDir = (fs::path(cfg.paths.dataDir) / "left").string();
    src->rightDir = (fs::path(cfg.paths.dataDir) / "right").string();

    // —— 【150FPS·方案 B】raw 模式分支：meta.json 存在 → 整载两 bin＋帧号表 ——
    const fs::path metaPath = fs::path(cfg.paths.dataDir) / "meta.json";
    std::error_code ecRaw;
    if (fs::exists(metaPath, ecRaw)) {
        std::ifstream mf(metaPath);
        if (!mf) {
            err = "meta.json 不可读: " + metaPath.string();
            return nullptr;
        }
        nlohmann::json meta;
        try {
            mf >> meta;
        } catch (const std::exception& e) {
            err = std::string("meta.json 解析失败: ") + e.what();
            return nullptr;
        }
        src->rawW = meta.value("width", 0);
        src->rawH = meta.value("height", 0);
        if (src->rawW <= 0 || src->rawH <= 0) {
            err = "meta.json 宽高非法";
            return nullptr;
        }
        for (int id : meta.value("frames", std::vector<int>{}))
            src->frameIds.push_back(id);
        std::sort(src->frameIds.begin(), src->frameIds.end());
        if (src->frameIds.empty()) {
            err = "meta.json 帧号表为空";
            return nullptr;
        }
        const size_t frameBytes = static_cast<size_t>(src->rawW) * src->rawH;
        const size_t total = frameBytes * src->frameIds.size();
        const char* names[2] = {"left.bin", "right.bin"};
        for (int side = 0; side < 2; ++side) {
            const fs::path bin = fs::path(cfg.paths.dataDir) / names[side];
            std::error_code ecs;
            const auto sz = fs::file_size(bin, ecs);
            if (ecs || sz != total) {
                err = names[side] + std::string(" 尺寸不符: ") + std::to_string(sz)
                      + " (期望 " + std::to_string(total) + ")";
                return nullptr;
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int side = 0; side < 2; ++side) {
            const fs::path bin = fs::path(cfg.paths.dataDir) / names[side];
            std::ifstream f(bin, std::ios::binary);
            if (!f) {
                err = "bin 打开失败: " + bin.string();
                return nullptr;
            }
            src->rawBuf[side].resize(total);
            if (!f.read(reinterpret_cast<char*>(src->rawBuf[side].data()),
                        static_cast<std::streamsize>(total))) {
                err = "bin 读取失败: " + bin.string();
                return nullptr;
            }
        }
        const double loadSec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        src->rawMode = true;
        // 【150FPS】锁页注册（JMW_NOPINRAW=1 关闭）：pageable H2D 上传经驱动
        // staging（实测 2.3-2.9GB/s 且第 2 笔恒慢——nsys 全部 Pageable 实锤）。
        // 【利用率专项·修复】三级降级注册，失败打印实因（原版静默吞错→横幅
        // pinned=off 但无人知晓原因）：ReadOnly 整块 → Default 整块 → 128MB 分块
        bool pinned = false;
        if (getenv("JMW_NOPINRAW") == nullptr) {
            pinned = true;
            for (int side = 0; side < 2; ++side) {
                unsigned char* base = src->rawBuf[side].data();
                cudaError_t e = cudaHostRegister(base, total, cudaHostRegisterReadOnly);
                if (e != cudaSuccess) {
                    fprintf(stderr, "[raw] cudaHostRegister(ReadOnly) side=%d (%zuMB): %s"
                                    " —— 改试 Default 整块\n",
                            side, total >> 20, cudaGetErrorString(e));
                    cudaGetLastError();
                    e = cudaHostRegister(base, total, cudaHostRegisterDefault);
                }
                if (e == cudaSuccess) {
                    src->rawReg[side].push_back({base, total});
                } else {
                    fprintf(stderr, "[raw] cudaHostRegister(Default) side=%d: %s"
                                    " —— 改试 128MB 分块\n",
                            side, cudaGetErrorString(e));
                    cudaGetLastError();
                    const size_t kChunk = size_t(128) << 20;
                    for (size_t off = 0; off < total; off += kChunk) {
                        const size_t n = std::min(kChunk, total - off);
                        cudaError_t ce = cudaHostRegister(base + off, n,
                                                          cudaHostRegisterDefault);
                        if (ce != cudaSuccess) {
                            fprintf(stderr, "[raw] 分块注册 side=%d off=%zuMB 失败: %s"
                                            " —— 本侧放弃（回滚已注册块）\n",
                                    side, off >> 20, cudaGetErrorString(ce));
                            cudaGetLastError();
                            for (auto& rg : src->rawReg[side])
                                cudaHostUnregister(rg.first);
                            src->rawReg[side].clear();
                            break;
                        }
                        src->rawReg[side].push_back({base + off, n});
                    }
                }
                src->rawPinned[side] = !src->rawReg[side].empty();
                if (!src->rawPinned[side]) pinned = false;
            }
        }
        std::cout << "[raw] 预载完成 " << src->frameIds.size() << " 帧 ×"
                  << src->rawW << "x" << src->rawH << " ("
                  << (total / 1024.0 / 1024.0) << "MB/眼, " << loadSec
                  << "s, pinned=" << (pinned ? "OK" : "off") << ")" << std::endl;
    } else {
        // —— PNG 模式：枚举 left/（right 必须对称，不对称报错）——
        std::set<int> ids;
        std::error_code ec;
        for (const auto& e : fs::directory_iterator(src->leftDir, ec)) {
            if (!e.is_regular_file()) continue;
            const std::string name = e.path().filename().string();
            if (name.size() != 10 || name.compare(name.size() - 4, 4, ".png") != 0)
                continue;
            try {
                ids.insert(std::stoi(name.substr(0, 6)));
            } catch (...) {
            }
        }
        if (ec) {
            err = "left 目录不可读: " + src->leftDir;
            return nullptr;
        }
        if (ids.empty()) {
            err = "left 目录无 NNNNNN.png 帧: " + src->leftDir;
            return nullptr;
        }
        for (int id : ids) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%06d", id);
            if (!fs::exists(fs::path(src->rightDir) / (std::string(buf) + ".png"), ec)) {
                err = std::string("right 缺对应帧: ") + buf + ".png";
                return nullptr;
            }
        }
        src->frameIds.assign(ids.begin(), ids.end());
    }

    // —— 奇偶＝激光管号（用户裁定 2026-10-04）——
    // 所有帧同时含激光＋标记点；奇偶交替是两支激光管轮流发射（不同管需调用
    // 不同标定参数——见 harness 按管注入激光温表）。亮度差异仅作两管形态的
    // 信息性记录（不做任何帧型判别/跳帧）。
    {
        int oddMore = 0, evenMore = 0, noVote = 0;
        const int pairs = std::min<int>(3, static_cast<int>(src->frameIds.size()) / 2);
        for (int i = 0; i < pairs; ++i) {
            const int a = src->frameIds[2 * i];      // 偶（管 B）
            const int b = src->frameIds[2 * i + 1];  // 奇（管 A）
            cv::Mat ga, gb;
            std::string e2;
            if (!src->readFrame(a, true, ga, e2) || !src->readFrame(b, true, gb, e2)) {
                ++noVote;
                continue;
            }
            const long ca = countBrightPixels(ga);
            const long cb = countBrightPixels(gb);
            if (ca < 0 || cb < 0 || std::min(ca, cb) == 0
                || std::abs(ca - cb) * 20 < std::min(ca, cb)) {
                ++noVote;
                continue;
            }
            if (cb > ca) ++oddMore; else ++evenMore;
        }
        std::cout << "[tube] 两管亮度投票（信息性）: 奇号管亮=" << oddMore
                  << " 偶号管亮=" << evenMore << " 弃权=" << noVote << std::endl;
    }

    // —— 配 cycle（同帧模式·全帧，用户裁定 2026-10-04）——
    // 每帧【同时含激光＋标记点】→ 逐帧各成一个 cycle（laserIdx = markerIdx = 同帧）：
    // 扫描仪运动中相邻帧位姿不同，激光 3D 与 R/T 必须出自同一帧同一位姿。
    // 奇偶交替＝两支激光管轮流发射（tubeParity 标记，供按管调用不同标定参数），
    // 不是"激光帧/补光帧"之分——不存在纯补光帧。
    for (int id : src->frameIds)
        src->cycles.push_back({id, id, (id % 2 == 1)});
    std::cout << "[frame] cycles=" << src->cycles.size() << "（逐帧同帧模式，奇偶＝激光管号）"
              << std::endl;
    return src;
}

// ============================================================================
// convertToRaw — PNG 目录 → raw 目录一次性转换（【150FPS·方案 B】）
// 输出：<dst>/left.bin＋right.bin（无头逐帧连续，字节＝imread 解码结果）＋
//       meta.json（width/height/frames 帧号表）
// ============================================================================
bool convertToRaw(const std::string& srcDir, const std::string& dstDir, std::string& err) {
    namespace fs = std::filesystem;
    const fs::path srcL = fs::path(srcDir) / "left";
    const fs::path srcR = fs::path(srcDir) / "right";
    std::error_code ec;
    if (!fs::exists(srcL, ec) || !fs::exists(srcR, ec)) {
        err = "源目录缺 left/ 或 right/: " + srcDir;
        return false;
    }
    // 枚举左目帧号（right 对称校验）
    std::set<int> ids;
    for (const auto& e : fs::directory_iterator(srcL, ec)) {
        const std::string name = e.path().filename().string();
        if (name.size() != 10 || name.compare(name.size() - 4, 4, ".png") != 0) continue;
        try { ids.insert(std::stoi(name.substr(0, 6))); } catch (...) {}
    }
    if (ids.empty()) {
        err = "源目录 left/ 无 NNNNNN.png: " + srcL.string();
        return false;
    }
    int W = 0, H = 0;
    for (int id : ids) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%06d", id);
        if (!fs::exists(srcR / (std::string(buf) + ".png"), ec)) {
            err = std::string("right 缺对应帧: ") + buf + ".png";
            return false;
        }
    }
    fs::create_directories(dstDir, ec);
    std::ofstream fL(fs::path(dstDir) / "left.bin", std::ios::binary);
    std::ofstream fR(fs::path(dstDir) / "right.bin", std::ios::binary);
    if (!fL || !fR) {
        err = "目标 bin 打开失败: " + dstDir;
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    int done = 0;
    for (int id : ids) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%06d", id);
        for (int side = 0; side < 2; ++side) {
            const fs::path p = (side == 0 ? srcL : srcR)
                               / (std::string(buf) + ".png");
            cv::Mat img = cv::imread(p.string(), cv::IMREAD_GRAYSCALE);
            if (img.empty() || img.type() != CV_8UC1) {
                err = "帧解码失败: " + p.string();
                return false;
            }
            if (W == 0) { W = img.cols; H = img.rows; }
            else if (img.cols != W || img.rows != H) {
                err = "帧尺寸不一致: " + p.string();
                return false;
            }
            auto& f = (side == 0 ? fL : fR);
            f.write(reinterpret_cast<const char*>(img.data),
                    static_cast<std::streamsize>(W) * H);
        }
        if (++done % 100 == 0)
            std::cout << "\r[to-raw] " << done << "/" << ids.size() << std::flush;
    }
    fL.close();
    fR.close();
    if (!fL || !fR) {
        err = "bin 写入失败（磁盘满?）: " + dstDir;
        return false;
    }
    nlohmann::json meta;
    meta["width"] = W;
    meta["height"] = H;
    meta["frames"] = std::vector<int>(ids.begin(), ids.end());
    {
        std::ofstream mf(fs::path(dstDir) / "meta.json");
        if (!mf) { err = "meta.json 写失败"; return false; }
        mf << meta.dump();
    }
    const double sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "\n[to-raw] 完成: " << ids.size() << " 帧 " << W << "x" << H
              << " → " << dstDir << " (" << sec << "s)" << std::endl;
    return true;
}

} // namespace scr
