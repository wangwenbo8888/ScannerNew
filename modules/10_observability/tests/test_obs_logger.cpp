// ============================================================================
// test_obs_logger.cpp — ObsLogger 日志装配/轮转/幂等/排障包单测（P3-T9，TDD 先行）
//
// 设计基准：docs/plans/2026-08-20-可观测性模块10设计方案.md §5
//   · init：控制台 + rotating_file（10MB×5）组 default logger；测试用 1KB×3
//     小容量加速验证轮转（审核 I 项）
//   · 幂等：重复 init 先拆旧装配再重建（第二个目录生效）；shutdown → init
//     再入路径同样覆盖
//   · 排障包：logs 副本 + version.txt 收进 logs 目录旁 diagnostics_<ts>/；
//     dump 路径空或不存在 → 跳过 dump 项不报错（容错）
//
// ⚠ spdlog default logger 是进程级全局：各用例用独立 tmp 目录隔离；
//   TearDown 必须先 obsLoggerShutdown()（释放 rotating sink 的 jmw.log 句柄）
//   再删目录——Windows 下句柄未释放删除必失败。
// ============================================================================
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/sink.h>

#include "ObsLogger.h"
#include "jmw_logging.h"

namespace fs = std::filesystem;

using Scanner::service::ObsLoggerConfig;
using Scanner::service::obsExportDiagnosticsPackage;
using Scanner::service::obsLoggerInit;
using Scanner::service::obsLoggerShutdown;

namespace {

class ObsLoggerTest : public ::testing::Test {
protected:
    std::string tmpRoot;   // ./test_obs_logger_tmp_<用例名>（相对路径，可重复跑）

    void SetUp() override {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        tmpRoot = std::string("./test_obs_logger_tmp_") + info->name();
        std::error_code ec;
        fs::remove_all(tmpRoot, ec);   // 用例开头若残留先删
    }

    void TearDown() override {
        obsLoggerShutdown();           // 先释放文件句柄，否则 Windows 删目录失败
        std::error_code ec;
        fs::remove_all(tmpRoot, ec);
    }

    // 小容量加速版装配：1KB×3（maxFiles=3 指含当前文件共保留 3 份）
    ObsLoggerConfig cfgFor(const std::string& sub) const {
        return ObsLoggerConfig{tmpRoot + "/" + sub, 1024, 3, "jmw"};
    }

    static std::string readAll(const fs::path& p) {
        std::ifstream in(p, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in),
                           std::istreambuf_iterator<char>());
    }

    static int countLogFiles(const fs::path& dir) {
        int n = 0;
        std::error_code ec;
        for (fs::directory_iterator it(dir, ec), end; it != end; it.increment(ec)) {
            if (it->is_regular_file(ec) && it->path().extension() == ".log") ++n;
        }
        return n;
    }
};

TEST_F(ObsLoggerTest, InitWritesTaggedLine) {
    obsLoggerInit(cfgFor("logs"));
    JMW_LOG_INFO("10-Test", "x={}", 1);
    spdlog::default_logger()->flush();

    const fs::path logFile = fs::path(tmpRoot) / "logs" / "jmw.log";
    ASSERT_TRUE(fs::exists(logFile)) << "init 后应产出 jmw.log";
    const std::string content = readAll(logFile);
    EXPECT_NE(content.find("[10-Test] x=1"), std::string::npos) << "消息须带 [模块-对象] 前缀";
    EXPECT_NE(content.find("[jmw]"), std::string::npos) << "pattern 含 logger 名（§5 格式）";
}

TEST_F(ObsLoggerTest, RotationKeepsMaxFiles) {
    obsLoggerInit(cfgFor("logs"));
    // 静音控制台 sink（sink[0]=stdout_color）：200×1KB 刷屏会污染 ctest 输出
    spdlog::default_logger()->sinks()[0]->set_level(spdlog::level::off);

    const std::string big(1000, 'x');   // 单行 ≈ 26(格式头) + 9(tag) + 1000 > 1KB
    for (int i = 0; i < 200; ++i) {
        JMW_LOG_INFO("10-Rot", "{}", big);
    }
    spdlog::default_logger()->flush();

    const fs::path logsDir = fs::path(tmpRoot) / "logs";
    EXPECT_EQ(countLogFiles(logsDir), 3) << "maxFiles=3 → 目录内恰好共 3 份 .log";
    EXPECT_TRUE(fs::exists(logsDir / "jmw.log"));
    EXPECT_TRUE(fs::exists(logsDir / "jmw.1.log"));
    EXPECT_TRUE(fs::exists(logsDir / "jmw.2.log"));
    EXPECT_FALSE(fs::exists(logsDir / "jmw.3.log"));
}

TEST_F(ObsLoggerTest, InitIdempotent) {
    // 路径①：不 shutdown 直接再 init（内部先拆旧装配）
    obsLoggerInit(cfgFor("logs1"));
    JMW_LOG_INFO("10-Idem", "first={}", 1);

    obsLoggerInit(cfgFor("logs2"));
    JMW_LOG_INFO("10-Idem", "second={}", 2);
    spdlog::default_logger()->flush();

    const std::string c1 = readAll(fs::path(tmpRoot) / "logs1" / "jmw.log");
    EXPECT_NE(c1.find("first=1"), std::string::npos) << "旧装配拆除时应落盘余量";

    const std::string c2 = readAll(fs::path(tmpRoot) / "logs2" / "jmw.log");
    ASSERT_FALSE(c2.empty()) << "第二个目录生效";
    EXPECT_NE(c2.find("second=2"), std::string::npos);

    // 路径②：shutdown 后再 init（进程内重建 default logger）
    obsLoggerShutdown();
    obsLoggerInit(cfgFor("logs3"));
    JMW_LOG_INFO("10-Idem", "third={}", 3);
    spdlog::default_logger()->flush();

    const std::string c3 = readAll(fs::path(tmpRoot) / "logs3" / "jmw.log");
    ASSERT_FALSE(c3.empty()) << "shutdown → init 再入后可正常写日志";
    EXPECT_NE(c3.find("third=3"), std::string::npos);
}

TEST_F(ObsLoggerTest, DiagnosticsPackage) {
    obsLoggerInit(cfgFor("logs"));
    JMW_LOG_INFO("10-Diag", "hello={}", 1);
    spdlog::default_logger()->flush();

    // dump 路径为空 → 跳过 dump 项不报错
    const std::string dst = obsExportDiagnosticsPackage("");
    EXPECT_FALSE(dst.empty());
    ASSERT_TRUE(fs::is_directory(fs::path(dst)));
    EXPECT_TRUE(fs::equivalent(fs::path(dst).parent_path(), fs::path(tmpRoot)))
        << "排障包目录产出在 logs 目录旁";
    ASSERT_TRUE(fs::exists(fs::path(dst) / "logs" / "jmw.log")) << "内含日志文件副本";
    EXPECT_NE(readAll(fs::path(dst) / "logs" / "jmw.log").find("hello=1"), std::string::npos);
    ASSERT_TRUE(fs::exists(fs::path(dst) / "version.txt")) << "内含版本信息文件";
    EXPECT_FALSE(fs::exists(fs::path(dst) / "dumps")) << "空 dump 路径 → 无 dump 项";

    // 容错：传入不存在的 dump 路径同样不报错
    EXPECT_NO_THROW(obsExportDiagnosticsPackage(tmpRoot + "/no_such_dump.dmp"));
}

} // namespace
