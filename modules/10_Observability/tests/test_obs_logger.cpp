// ============================================================================
// test_obs_logger.cpp — ObsLogger 日志装配/按日文件/幂等/排障包单测（P3-T9）
//
// 设计基准：docs/plans/2026-08-20-可观测性模块10设计方案.md §5（按日改造）
//   · init：控制台 + daily_file 组 default logger；文件 jmw_<今日>.log
//     （日期插扩展名前，spdlog calc_filename 语义）
//   · 保留清理：预置多份伪历史日期文件（ISO 日期字典序=时间序），init 后
//     超 maxFiles 的最旧被删（sink 构造时即清一次）
//   · 幂等：重复 init 先拆旧装配再重建（第二个目录生效）；shutdown → init
//     再入路径同样覆盖
//   · 排障包：logs 副本 + version.txt 收进 logs 目录旁 diagnostics_<ts>/；
//     dump 路径空或不存在 → 跳过 dump 项不报错（容错）
//
// ⚠ spdlog default logger 是进程级全局：各用例用独立 tmp 目录隔离；
//   TearDown 必须先 obsLoggerShutdown()（释放 daily sink 的当日文件句柄）
//   再删目录——Windows 下句柄未释放删除必失败。
// ⚠ 跨零点切换属 sink 内部时钟行为，不在此测（单测无法注入时钟）。
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

    // 小份数加速版装配：保留 3 份（含当天）
    ObsLoggerConfig cfgFor(const std::string& sub) const {
        return ObsLoggerConfig{tmpRoot + "/" + sub, 3, "jmw"};
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

    // 今日日期串（spdlog calc_filename 同款格式 YYYY-MM-DD，本地时区）
    static std::string todayStr() {
        const std::time_t t = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        std::tm tm{};
        localtime_s(&tm, &t);
        char buf[16];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm);
        return buf;
    }

    static void touch(const fs::path& p) {
        fs::create_directories(p.parent_path());
        std::ofstream(p) << 'x';
    }
};

TEST_F(ObsLoggerTest, InitWritesTaggedLineToDatedFile) {
    obsLoggerInit(cfgFor("logs"));
    JMW_LOG_INFO("10-Test", "x={}", 1);
    spdlog::default_logger()->flush();

    const fs::path logFile =
        fs::path(tmpRoot) / "logs" / ("jmw_" + todayStr() + ".log");
    ASSERT_TRUE(fs::exists(logFile)) << "init 后应产出按日命名的 jmw_<今日>.log";
    const std::string content = readAll(logFile);
    EXPECT_NE(content.find("[10-Test] x=1"), std::string::npos) << "消息须带 [模块-对象] 前缀";
    EXPECT_NE(content.find("[jmw]"), std::string::npos) << "pattern 含 logger 名（§5 格式）";
    // 源位置：宏在调用点捕获——须含本测试文件名（basename）与行号分隔
    EXPECT_NE(content.find("test_obs_logger.cpp:"), std::string::npos)
        << "pattern %s:%# 应输出源文件名:行号（JMW_LOG 宏族自动捕获）";
}

TEST_F(ObsLoggerTest, InitCleanupKeepsMaxFiles) {
    const fs::path logsDir = fs::path(tmpRoot) / "logs";
    // 预置 4 份伪历史日期文件（字典序=时间序）＋init 产出今日一份 = 5 份候选
    touch(logsDir / "jmw_2000-01-01.log");
    touch(logsDir / "jmw_2000-01-02.log");
    touch(logsDir / "jmw_2000-01-03.log");
    touch(logsDir / "jmw_2000-01-04.log");

    obsLoggerInit(cfgFor("logs"));   // maxFiles=3：sink 构造时清一次最旧
    JMW_LOG_INFO("10-Cln", "keep={}", 1);
    spdlog::default_logger()->flush();

    EXPECT_EQ(countLogFiles(logsDir), 3) << "maxFiles=3 → 目录内恰好共 3 份 .log";
    EXPECT_FALSE(fs::exists(logsDir / "jmw_2000-01-01.log")) << "最旧被删";
    EXPECT_FALSE(fs::exists(logsDir / "jmw_2000-01-02.log")) << "次旧被删";
    EXPECT_TRUE(fs::exists(logsDir / "jmw_2000-01-03.log")) << "保留较新历史";
    EXPECT_TRUE(fs::exists(logsDir / "jmw_2000-01-04.log"));
    EXPECT_TRUE(fs::exists(logsDir / ("jmw_" + todayStr() + ".log"))) << "今日在册";
}

TEST_F(ObsLoggerTest, InitIdempotent) {
    const std::string todayFile = "jmw_" + todayStr() + ".log";
    // 路径①：不 shutdown 直接再 init（内部先拆旧装配）
    obsLoggerInit(cfgFor("logs1"));
    JMW_LOG_INFO("10-Idem", "first={}", 1);

    obsLoggerInit(cfgFor("logs2"));
    JMW_LOG_INFO("10-Idem", "second={}", 2);
    spdlog::default_logger()->flush();

    const std::string c1 = readAll(fs::path(tmpRoot) / "logs1" / todayFile);
    EXPECT_NE(c1.find("first=1"), std::string::npos) << "旧装配拆除时应落盘余量";

    const std::string c2 = readAll(fs::path(tmpRoot) / "logs2" / todayFile);
    ASSERT_FALSE(c2.empty()) << "第二个目录生效";
    EXPECT_NE(c2.find("second=2"), std::string::npos);

    // 路径②：shutdown 后再 init（进程内重建 default logger）
    obsLoggerShutdown();
    obsLoggerInit(cfgFor("logs3"));
    JMW_LOG_INFO("10-Idem", "third={}", 3);
    spdlog::default_logger()->flush();

    const std::string c3 = readAll(fs::path(tmpRoot) / "logs3" / todayFile);
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
    const fs::path logCopy =
        fs::path(dst) / "logs" / ("jmw_" + todayStr() + ".log");
    ASSERT_TRUE(fs::exists(logCopy)) << "内含当日日志文件副本";
    EXPECT_NE(readAll(logCopy).find("hello=1"), std::string::npos);
    ASSERT_TRUE(fs::exists(fs::path(dst) / "version.txt")) << "内含版本信息文件";
    EXPECT_FALSE(fs::exists(fs::path(dst) / "dumps")) << "空 dump 路径 → 无 dump 项";

    // 容错：传入不存在的 dump 路径同样不报错
    EXPECT_NO_THROW(obsExportDiagnosticsPackage(tmpRoot + "/no_such_dump.dmp"));
}

} // namespace