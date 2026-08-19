// ============================================================================
// CalibComputePipeline.cpp — B 标定计算流水线对象（两线程总装）
// ============================================================================
// cam（0..50）∥ las（50..100）经 promise/future 衔接；promise 移交相机线程持有
// ——相机链早期失败（未兑现）时线程退出即析构 promise → 激光链得破诺快速失败。
// 输出合并：两链写 CalibComputeOutput 不同成员（相机：stereo/三表/quality；
// 激光：pjc/planeMap/planeMapTempTable/laserExtrinsicTempTable/laserValid），
// join 后总装层合成 quality。
#include "pipelines/calibcompute/CalibComputePipeline.h"

#include <memory>
#include <mutex>
#include <utility>

#include "pipelines/calibcompute/CameraChain.h"
#include "pipelines/calibcompute/LaserChain.h"

namespace Scanner::pipeline {

CalibComputePipeline::CalibComputePipeline(const Config& cfg) : cfg_(cfg) {}

CalibComputePipeline::~CalibComputePipeline() {
    stop();   // join 后台线程（避免析构 joinable 线程 terminate）
}

void CalibComputePipeline::attachBoardPoints(std::vector<cv::Point3f> board3D) {
    board_ = std::move(board3D);
}

void CalibComputePipeline::attachInitialParams(InitialCalibParams init) {
    init_ = std::move(init);
    hasInit_ = true;
}

void CalibComputePipeline::attachSession(PostureSessionData session) {
    session_ = std::move(session);
    hasSession_ = true;
}

Scanner::Result CalibComputePipeline::configure(const PipelineDeps& deps) {
    if (running_.load())
        return Scanner::Result::fail("cannot configure while running");
    calibRepo_ = deps.calibRepo;   // 落盘 T23 接线：run 尾写（本版仅记录）
    return Scanner::Result::ok("calib pipeline configured");
}

void CalibComputePipeline::setTestHooks(TestHooks hooks) {
    hooks_ = std::move(hooks);
}

const CalibComputeOutput& CalibComputePipeline::output() const { return out_; }

bool CalibComputePipeline::isRunning() const { return running_.load(); }

Scanner::Result CalibComputePipeline::run(const PostureSessionData& in,
                                          const ProgressCb& cb,
                                          CancelToken& cancel) {
    std::lock_guard<std::mutex> lock(runMutex_);
    if (running_.load())
        return Scanner::Result::fail("calib compute already running");
    return runLocked(in, cb, cancel);
}

Scanner::Result CalibComputePipeline::runLocked(const PostureSessionData& in,
                                                const ProgressCb& cb,
                                                CancelToken& cancel) {
    running_.store(true);
    struct RunningGuard {
        std::atomic<bool>& flag;
        ~RunningGuard() { flag.store(false); }
    } runningGuard{running_};

    // —— 链装配（测试钩子优先；缺省真链）——
    CameraRunFn camRun = hooks_.cameraRun;
    LaserRunFn lasRun = hooks_.laserRun;
    if (!camRun) {
        if (!hasInit_ || board_.empty())
            return Scanner::Result::fail(
                "camera chain requires attachInitialParams + attachBoardPoints");
        camRun = [](const PostureSessionData& in2, const InitialCalibParams& init2,
                    const std::vector<cv::Point3f>& board2, StereoParams& outStereo,
                    std::promise<StereoParams>& toLaser, CalibComputeOutput& out2,
                    const ProgressCb& cb2, const CancelToken& cancel2) {
            CameraChain chain;
            return chain.run(in2, init2, board2, outStereo, toLaser, out2, cb2, cancel2);
        };
    }
    if (!lasRun) {
        auto chain = std::make_shared<LaserChain>(
            LaserChain::Deps{cfg_.pjcInitialT, LaserOps{}});
        lasRun = [chain](const PostureSessionData& in2, std::future<StereoParams> fut,
                         CalibComputeOutput& out2, const ProgressCb& cb2,
                         const CancelToken& cancel2) {
            return chain->run(in2, std::move(fut), out2, cb2, cancel2);
        };
    }

    out_ = CalibComputeOutput{};   // 重置上次输出

    // —— 进度合并：单调门（两链本域 0..50/50..100；仅转发更大值）——
    if (cb) cb(0, "calib compute start");
    std::mutex pctMutex;
    int lastPct = 0;
    auto merged = [&cb, &pctMutex, &lastPct](int p, const std::string& s) {
        if (!cb) return;
        std::lock_guard<std::mutex> lk(pctMutex);
        if (p > lastPct) {
            lastPct = p;
            cb(p, s);
        }
    };

    // promise 移交相机线程：相机链早期失败不兑现 → 线程退出析构 → 破诺唤醒激光链
    auto promiseHolder = std::make_unique<std::promise<StereoParams>>();
    auto fut = promiseHolder->get_future();

    Scanner::Result camRes = Scanner::Result::fail("camera chain not run");
    Scanner::Result lasRes = Scanner::Result::fail("laser chain not run");
    StereoParams outStereo;   // 相机链契约出参（out_.stereo 为总装输出面）

    std::thread camThread([this, &in, camRun, &outStereo, &merged, &cancel,
                           ph = std::move(promiseHolder), &camRes]() {
        std::promise<StereoParams> promiseOwned = std::move(*ph);
        try {
            camRes = camRun(in, init_, board_, outStereo, promiseOwned, out_, merged, cancel);
        } catch (const std::exception& e) {
            camRes = Scanner::Result::fail(std::string("camera chain thread exception: ") + e.what());
        } catch (...) {
            camRes = Scanner::Result::fail("camera chain thread unknown exception");
        }
    });
    std::thread lasThread([this, &in, lasRun, &lasRes, &merged, &cancel,
                           f = std::move(fut)]() mutable {
        try {
            lasRes = lasRun(in, std::move(f), out_, merged, cancel);
        } catch (const std::exception& e) {
            lasRes = Scanner::Result::fail(std::string("laser chain thread exception: ") + e.what());
        } catch (...) {
            lasRes = Scanner::Result::fail("laser chain thread unknown exception");
        }
    });

    camThread.join();
    lasThread.join();

    // —— 合成（join 后写 quality，无跨线程竞争）——
    if (camRes.success && lasRes.success) {
        out_.quality.ok = true;
        out_.quality.summary = "calib compute ok | camera: " + camRes.message +
                               " | laser: " + lasRes.message;
        lastResult_ = Scanner::Result::ok(out_.quality.summary);
    } else {
        out_.quality.ok = false;
        out_.quality.summary = "calib compute fail | camera(ok=" +
                               std::string(camRes.success ? "1" : "0") + "): " + camRes.message +
                               " | laser(ok=" + std::string(lasRes.success ? "1" : "0") +
                               "): " + lasRes.message;
        lastResult_ = Scanner::Result::fail(out_.quality.summary);
    }
    if (cb) {
        std::lock_guard<std::mutex> lk(pctMutex);
        cb(100, "calib compute done");   // 收尾必报 100（单调门终点）
    }
    return lastResult_;
}

// —— IPipelineObject 适配：start=后台 run（attachSession 会话）；stop=取消+join ——
Scanner::Result CalibComputePipeline::start() {
    std::lock_guard<std::mutex> lock(runMutex_);
    if (running_.load())
        return Scanner::Result::fail("calib compute already running");
    if (!hasSession_)
        return Scanner::Result::fail("start() requires attachSession(...) first");
    if (worker_.joinable()) worker_.join();

    auto token = std::make_shared<CancelToken>();
    cancelToken_ = token;                   // 管道持有；worker 捕获共享所有权（保活至线程退出）
    PostureSessionData snapshot = session_; // 快照（主线程不再动会话）
    running_.store(true);                   // 先置位（runLocked 的 guard 负责清位）
    try {
        worker_ = std::thread([this, token, snap = std::move(snapshot)]() mutable {
            runLocked(snap, nullptr, *token);
        });
    } catch (...) {                         // 线程构造失败：复位标志/令牌，不留卡死态
        running_.store(false);
        cancelToken_.reset();
        return Scanner::Result::fail("calib compute failed to start worker thread");
    }
    return Scanner::Result::ok("calib compute started (background run)");
}

void CalibComputePipeline::stop() {
    // 锁内取令牌共享引用再 cancel：与 start() 重建令牌互斥（消 UAF 窗口）
    std::shared_ptr<CancelToken> token;
    {
        std::lock_guard<std::mutex> lock(runMutex_);
        token = cancelToken_;
    }
    if (token) token->cancel();
    std::thread local;
    {
        std::lock_guard<std::mutex> lock(runMutex_);
        local = std::move(worker_);
    }
    if (local.joinable()) local.join();
}

} // namespace Scanner::pipeline
