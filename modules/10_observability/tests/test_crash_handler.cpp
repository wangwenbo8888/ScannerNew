// ============================================================================
// test_crash_handler.cpp — CrashHandler SEH 捕获/残留检测单测（P3-T10，TDD 先行）
//
// 设计基准：docs/plans/2026-08-20-可观测性模块10设计方案.md §7
//   · filter 内只做最小安全动作：MiniDumpWriteDump + crash_pending 标记
//     （禁堆分配/锁/spdlog/zip——SEH filter 内不安全，§7.1）
//   · 崩溃用例独立子进程：本 exe 带 --crash-child <dir> 再入——install 后
//     主动解引用空指针；父进程 std::system 取退出码并断言 dump/标记落盘
//   · 残留检测：检出后不删除（打包/上报归调用方，app T12 接线时处置）
//
// ⚠ 本文件自带 main（--crash-child 分支须抢在 gtest 之前），CMake 对该
//   target 特判链 GTest::gtest 而非 gtest_main。
// ============================================================================
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "CrashHandler.h"

namespace fs = std::filesystem;

using Scanner::service::crash::detectResidualDump;
using Scanner::service::crash::install;

namespace {

std::string g_selfExe;   // main 里取 argv[0] 转绝对路径（ctest 下子进程命令须绝对路径）

// 用例级临时目录：开头清残留；测试 exe 工作目录=构建目录
std::string freshDir(const char* name) {
    const std::string dir = std::string("./test_crash_handler_tmp_") + name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    return dir;
}

void quietRemove(const std::string& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
}

void writeByteFile(const fs::path& p) {
    std::ofstream out(p, std::ios::binary);
    out << 'x';
}

int countDmpFiles(const fs::path& dir) {
    int n = 0;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; it != end; it.increment(ec)) {
        if (it->is_regular_file(ec) && it->path().extension() == ".dmp") ++n;
    }
    return n;
}

// 起子进程崩溃样本：返回退出码（崩溃=异常码，非 0）。
// ⚠ cmd.exe /c 引号规则：命令行多于两个引号时首尾引号会被剥掉（§2 老行为），
// 外面再包一层引号护住内层引号，路径含空格也稳
int runCrashChild(const std::string& dir) {
    const std::string cmd = "\"\"" + g_selfExe + "\" --crash-child \"" + dir + "\"\"";
    return std::system(cmd.c_str());
}

} // namespace

TEST(Crash, InstallCreatesDirAndReturnsTrue) {
    const std::string dir = freshDir("install") + "/dumps";   // 多级路径验证 create_directories
    EXPECT_FALSE(fs::exists(dir)) << "前置：目录不存在";
    EXPECT_TRUE(install(dir));
    EXPECT_TRUE(fs::is_directory(dir)) << "install 应建好 dump 目录";
    EXPECT_TRUE(install(dir)) << "重复 install 幂等无害";
    quietRemove(freshDir("install"));
}

TEST(Crash, ChildProcessCrashProducesDump) {
    const std::string dir = freshDir("dump");
    ASSERT_FALSE(g_selfExe.empty());
    const int rc = runCrashChild(dir);
    EXPECT_NE(rc, 0) << "子进程崩溃退出码应非 0";
    EXPECT_EQ(countDmpFiles(dir), 1) << "崩溃后目录内应出现 1 份 .dmp";
    quietRemove(freshDir("dump"));
}

TEST(Crash, MarkFileWritten) {
    const std::string dir = freshDir("mark");
    ASSERT_NE(runCrashChild(dir), 0);
    EXPECT_TRUE(fs::exists(fs::path(dir) / "crash_pending"))
        << "filter 应写 crash_pending 标记文件（§7.1 最小动作）";
    quietRemove(freshDir("mark"));
}

TEST(Crash, DetectResidualDump) {
    const std::string dir = freshDir("residual");
    ASSERT_TRUE(install(dir));

    std::string out;
    EXPECT_FALSE(detectResidualDump(out)) << "空目录无残留";

    const fs::path older = fs::path(dir) / "crash_100.dmp";
    const fs::path newer = fs::path(dir) / "crash_200.dmp";
    writeByteFile(older);
    writeByteFile(newer);
    // 手工设写时间保证新旧次序确定（同秒创建可能并列）
    std::error_code ec;
    const auto tOld = fs::file_time_type::clock::now() - std::chrono::hours(2);
    fs::last_write_time(older, tOld, ec);
    fs::last_write_time(newer, tOld + std::chrono::hours(1), ec);

    EXPECT_TRUE(detectResidualDump(out)) << "有 .dmp → 检出残留";
    ASSERT_FALSE(out.empty());
    EXPECT_TRUE(fs::equivalent(fs::path(out), older)) << "应返回最旧一份";
    EXPECT_TRUE(fs::exists(older)) << "检出后不删除（打包归调用方）";

    fs::remove(older, ec);
    fs::remove(newer, ec);
    EXPECT_FALSE(detectResidualDump(out)) << "清空后无残留";
    quietRemove(freshDir("residual"));
}

// ============================================================================
// 自带 main：--crash-child <dumpDir> 分支 = 子进程崩溃样本（install → 空指针
// 写触发 SEH → filter 落 dump+标记 → 进程以异常码退出）；否则正常跑 gtest。
// ============================================================================
int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--crash-child") == 0) {
        install(argv[2]);
        *(volatile int*)nullptr = 42;   // 主动崩溃（volatile 防编译器优化掉写入）
        return 0;                       // 不可达：filter 返回 EXECUTE_HANDLER 后进程以异常码终止
    }
    if (argv[0] && *argv[0]) g_selfExe = fs::absolute(fs::path(argv[0])).string();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
