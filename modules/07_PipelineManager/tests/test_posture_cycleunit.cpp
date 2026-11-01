#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>
#include <opencv2/core.hpp>
#include "CycleUnit.h"
#include "SlotRing.h"
using Scanner::data::CycleUnit;
using Scanner::data::SlotRing;

namespace {

constexpr int kMarkerSize = 64;    // 标记点帧 64×64
constexpr int kLaserW = 32;        // 激光"乘客"帧 32×32
constexpr int kLaserFrames = 8;    // N*2 = 8 张（约定偶=L / 奇=R）

// 同尺寸同类型且逐像素相等
bool matEq(const cv::Mat& a, const cv::Mat& b) {
    return a.size() == b.size() && a.type() == b.type()
        && cv::countNonZero(a != b) == 0;
}

// CV_8UC1 全图恒为 v
bool allEq(const cv::Mat& m, cv::Scalar v) {
    return cv::countNonZero(m != v) == 0;
}

// 随机内容单元（与原始引用比对，不依赖可复现随机源）
std::shared_ptr<CycleUnit> mkRandomUnit(uint64_t id, double temperature) {
    auto u = std::make_shared<CycleUnit>();
    u->id = id;
    cv::RNG rng(static_cast<uint64_t>(id) * 7919 + 13);
    u->markerL = cv::Mat(kMarkerSize, kMarkerSize, CV_8UC1);
    u->markerR = cv::Mat(kMarkerSize, kMarkerSize, CV_8UC1);
    rng.fill(u->markerL, cv::RNG::UNIFORM, 0, 256);
    rng.fill(u->markerR, cv::RNG::UNIFORM, 0, 256);
    u->laserFrames.resize(kLaserFrames);
    for (auto& f : u->laserFrames) {
        f = cv::Mat(kLaserW, kLaserW, CV_8UC1);
        rng.fill(f, cv::RNG::UNIFORM, 0, 256);
    }
    u->temperature = temperature;
    u->timestamp = 1000 + id;
    return u;
}

// 恒定内容单元（消费侧可按 id 重算期望值，验证无丢失、不错帧）
uchar baseVal(uint64_t id) { return static_cast<uchar>((id + 1) * 25 % 240); }  // 留 +9 余量防饱和截断
std::shared_ptr<CycleUnit> mkConstantUnit(uint64_t id) {
    auto u = std::make_shared<CycleUnit>();
    u->id = id;
    const uchar v = baseVal(id);
    u->markerL = cv::Mat(kMarkerSize, kMarkerSize, CV_8UC1, cv::Scalar(v));
    u->markerR = cv::Mat(kMarkerSize, kMarkerSize, CV_8UC1, cv::Scalar(v + 1));
    u->laserFrames.resize(kLaserFrames);
    for (int i = 0; i < kLaserFrames; ++i)
        u->laserFrames[i] = cv::Mat(kLaserW, kLaserW, CV_8UC1, cv::Scalar(v + 2 + i));
    u->temperature = 20.0 + static_cast<double>(id);
    u->timestamp = 5000 + id;
    return u;
}

} // namespace

// 语义 1：CycleUnit 经 SlotRing 写读回环——字段全等（Mat 按 data 指针 + 内容比对）
TEST(CycleUnit, CycleUnitRoundtrip) {
    SlotRing<CycleUnit> ring(4);
    auto src = mkRandomUnit(3, 25.1);
    ring.write(src);
    auto out = ring.read(0);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->id, 3u);
    EXPECT_DOUBLE_EQ(out->temperature, 25.1);
    EXPECT_EQ(out->timestamp, 1003u);
    EXPECT_EQ(out->d_markerL, nullptr);          // device 副本预留位原样
    EXPECT_EQ(out->d_markerR, nullptr);
    ASSERT_EQ(out->laserFrames.size(), static_cast<size_t>(kLaserFrames));
    EXPECT_EQ(out->markerL.data, src->markerL.data);   // 同一份数据，未拷贝
    EXPECT_EQ(out->markerR.data, src->markerR.data);
    EXPECT_TRUE(matEq(out->markerL, src->markerL));    // 且内容相等
    EXPECT_TRUE(matEq(out->markerR, src->markerR));
}

// 语义 2：Backpressure 端到端——4 槽写 10 单元：消费前写满阻塞，
//         消费者顺序 claim/read/done 全取，10 单元全到（id 0..9）无丢失
TEST(CycleUnit, BackpressureFullCycleKeepsAll) {
    SlotRing<CycleUnit> ring(4, SlotRing<CycleUnit>::WriterMode::Backpressure);
    constexpr uint64_t kTotal = 10;
    std::atomic<bool> produced{false};
    std::thread producer([&] {
        for (uint64_t id = 0; id < kTotal; ++id) ring.write(mkConstantUnit(id));
        produced = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));  // 让环写满、生产者阻塞在第 5 写
    EXPECT_FALSE(produced.load());       // 反压生效：不消费则写不完
    EXPECT_EQ(ring.writePtr(), 4u);      // 恰好写满 4 槽

    for (uint64_t id = 0; id < kTotal; ++id) {
        EXPECT_EQ(ring.claim(), id);     // 顺序领号
        ASSERT_TRUE(ring.waitFor(id, std::chrono::seconds(2)));
        auto u = ring.read(id);
        ASSERT_NE(u, nullptr);
        EXPECT_EQ(u->id, id);            // 顺序全到、无丢失
        const uchar v = baseVal(id);
        EXPECT_TRUE(allEq(u->markerL, cv::Scalar(v)));
        EXPECT_TRUE(allEq(u->markerR, cv::Scalar(v + 1)));
        ASSERT_EQ(u->laserFrames.size(), static_cast<size_t>(kLaserFrames));
        for (int i = 0; i < kLaserFrames; ++i)
            EXPECT_TRUE(allEq(u->laserFrames[i], cv::Scalar(v + 2 + i)));
        EXPECT_DOUBLE_EQ(u->temperature, 20.0 + static_cast<double>(id));
        ring.done();                     // 腾位唤醒生产者
    }
    producer.join();
    EXPECT_TRUE(produced.load());
    EXPECT_EQ(ring.writePtr(), kTotal);
    EXPECT_EQ(ring.donePtr(), kTotal);
}

// 语义 3：乘客帧完整性——读出后 laserFrames 张数与内容逐张比对不缺不坏
TEST(CycleUnit, PassengerFramesIntact) {
    SlotRing<CycleUnit> ring(4);
    auto src = mkRandomUnit(7, 30.0);
    ring.write(src);
    auto out = ring.read(0);
    ASSERT_NE(out, nullptr);
    ASSERT_EQ(out->laserFrames.size(), src->laserFrames.size());
    for (size_t i = 0; i < src->laserFrames.size(); ++i) {
        EXPECT_EQ(out->laserFrames[i].data, src->laserFrames[i].data);  // 未拷贝
        EXPECT_TRUE(matEq(out->laserFrames[i], src->laserFrames[i]));   // 内容逐张相等
    }
}

// 语义 4：持引用不悬空——覆盖旧槽后持引用者内容不变（大对象场景：8 张乘客帧全保活）
TEST(CycleUnit, DeepCopySemantics) {
    SlotRing<CycleUnit> ring(2, SlotRing<CycleUnit>::WriterMode::Overwrite);
    auto src = mkRandomUnit(0, 21.0);
    ring.write(src);
    auto kept = ring.read(0);                    // 覆盖前持引用
    ASSERT_NE(kept, nullptr);
    EXPECT_EQ(kept.get(), src.get());            // 同一对象（引用计数保活，非深拷贝）
    cv::Mat lRef = src->markerL.clone();         // 独立参照副本：数据悬空/被篡改即现形
    cv::Mat rRef = src->markerR.clone();
    std::vector<cv::Mat> laserRef(src->laserFrames.size());
    for (size_t i = 0; i < laserRef.size(); ++i) laserRef[i] = src->laserFrames[i].clone();

    ring.write(mkRandomUnit(1, 22.0));
    ring.write(mkRandomUnit(2, 23.0));           // 覆盖槽 0
    EXPECT_EQ(ring.read(0), nullptr);            // 新读者：确定已覆盖 → nullptr

    EXPECT_EQ(kept->id, 0u);
    EXPECT_DOUBLE_EQ(kept->temperature, 21.0);
    EXPECT_TRUE(matEq(kept->markerL, lRef));
    EXPECT_TRUE(matEq(kept->markerR, rRef));
    ASSERT_EQ(kept->laserFrames.size(), laserRef.size());
    for (size_t i = 0; i < laserRef.size(); ++i)
        EXPECT_TRUE(matEq(kept->laserFrames[i], laserRef[i]));  // 8 张乘客帧全不悬空
}
