// ============================================================================
// test_cloud_warehouse.cpp — 点云仓库（PointCloudBuffer）契约测试
//
// B-T4：marker 通道（续扫基准，globalId 保真）——setMarkers/snapshotMarkers
//       独立锁＋独立版本号（与点云版本分家），并发读写无死锁。
// B-T5：按需导出——exportCloud/exportMarkers 走 fileio（02-D5 唯一出口）。
// ============================================================================

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "PointCloudBuffer.h"
#include "file_io.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using Scanner::data::MarkerRecord;
using Scanner::data::PointCloudBuffer;
namespace fio = Scanner::data::fileio;
namespace fs = std::filesystem;

namespace {

fs::path tempFile(const char* name) {
    return fs::temp_directory_path() / name;
}

void expectMarkerEq(const MarkerRecord& got, const MarkerRecord& want) {
    EXPECT_EQ(got.globalId, want.globalId);
    EXPECT_FLOAT_EQ(got.pos.x, want.pos.x);
    EXPECT_FLOAT_EQ(got.pos.y, want.pos.y);
    EXPECT_FLOAT_EQ(got.pos.z, want.pos.z);
    EXPECT_FLOAT_EQ(got.normal[0], want.normal[0]);
    EXPECT_FLOAT_EQ(got.normal[1], want.normal[1]);
    EXPECT_FLOAT_EQ(got.normal[2], want.normal[2]);
}

} // namespace

// —— 1. marker 往返：globalId/pos/normal 全等 ——
TEST(CloudWarehouse, MarkersRoundtrip) {
    PointCloudBuffer buf;
    MarkerRecord in;
    in.globalId = 7;
    in.pos = cv::Point3f(1.f, 2.f, 3.f);
    in.normal = cv::Vec3f(0.f, 0.f, 1.f);
    buf.setMarkers({in});

    uint64_t v = 0;
    std::vector<MarkerRecord> out;
    buf.snapshotMarkers(v, out);
    ASSERT_EQ(out.size(), 1u);
    expectMarkerEq(out[0], in);
}

// —— 2. marker 版本号：未写过为 0，每次 setMarkers 递增 ——
TEST(CloudWarehouse, MarkersVersionBumps) {
    PointCloudBuffer buf;
    uint64_t v = 999;
    std::vector<MarkerRecord> out;
    buf.snapshotMarkers(v, out);
    EXPECT_EQ(v, 0u);
    EXPECT_TRUE(out.empty());

    MarkerRecord m;
    m.globalId = 1;
    m.pos = cv::Point3f(0.f, 0.f, 0.f);
    m.normal = cv::Vec3f(1.f, 0.f, 0.f);
    buf.setMarkers({m});
    buf.snapshotMarkers(v, out);
    EXPECT_EQ(v, 1u);

    buf.setMarkers({});
    buf.snapshotMarkers(v, out);
    EXPECT_EQ(v, 2u);
    EXPECT_TRUE(out.empty());
}

// —— 3. marker 通道并发读写：独立锁，~2s 无死锁无崩 ——
TEST(CloudWarehouse, ConcurrentMarkerWriteRead) {
    PointCloudBuffer buf;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> reads{0};

    std::thread writer([&] {
        uint32_t id = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            std::vector<MarkerRecord> ms(4);
            for (auto& m : ms) {
                m.globalId = id;
                m.pos = cv::Point3f(1.f, 2.f, 3.f);
                m.normal = cv::Vec3f(0.f, 0.f, 1.f);
            }
            buf.setMarkers(ms);
            ++id;
        }
    });
    std::thread reader([&] {
        uint64_t v = 0;
        std::vector<MarkerRecord> out;
        while (!stop.load(std::memory_order_relaxed)) {
            buf.snapshotMarkers(v, out);
            reads.fetch_add(1);
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));
    stop.store(true);
    writer.join();
    reader.join();
    EXPECT_GT(reads.load(), 0u);
}

// —— 4. 点云按需导出：pushPointCloud → exportCloud(.ply) → fileio 读回点数一致 ——
TEST(CloudWarehouse, ExportCloudRoundtrip) {
    PointCloudBuffer buf;
    Scanner::data::PointCloudFrame frame;
    frame.points = {{1.5f, 2.5f, 3.5f}, {-4.f, 5.f, -6.f}, {0.f, 100.f, 0.25f}, {7.f, 8.f, 9.f}};
    frame.pointCount = static_cast<int>(frame.points.size());
    ASSERT_TRUE(buf.pushPointCloud(frame).success);

    auto path = tempFile("jmw_cloud_warehouse.ply").string();
    ASSERT_TRUE(buf.exportCloud(path));

    std::vector<cv::Point3f> out;
    ASSERT_TRUE(fio::importPLY(path, out));
    EXPECT_EQ(out.size(), frame.points.size());
    fs::remove(path);
}

// —— 5. 标志点按需导出：setMarkers → exportMarkers(.json) → fileio 读回数量一致 ——
TEST(CloudWarehouse, ExportMarkersRoundtrip) {
    PointCloudBuffer buf;
    std::vector<MarkerRecord> ms(3);
    for (size_t i = 0; i < ms.size(); ++i) {
        ms[i].globalId = static_cast<uint32_t>(i);
        ms[i].pos = cv::Point3f(1.f * i, 2.f * i, 3.f * i);
        ms[i].normal = cv::Vec3f(0.f, 0.f, 1.f);
    }
    buf.setMarkers(ms);

    auto path = tempFile("jmw_cloud_warehouse_markers.json").string();
    ASSERT_TRUE(buf.exportMarkers(path));

    std::vector<cv::Point3f> out;
    ASSERT_TRUE(fio::importMarkers(path, out));
    EXPECT_EQ(out.size(), ms.size());
    fs::remove(path);
}

// —— 6. 空仓库导出：返回 true（空文件），不崩 ——
TEST(CloudWarehouse, ExportEmptyOk) {
    PointCloudBuffer buf;
    auto ply = tempFile("jmw_cloud_warehouse_empty.ply").string();
    auto json = tempFile("jmw_cloud_warehouse_empty.json").string();
    EXPECT_TRUE(buf.exportCloud(ply));
    EXPECT_TRUE(buf.exportMarkers(json));
    fs::remove(ply);
    fs::remove(json);
}
