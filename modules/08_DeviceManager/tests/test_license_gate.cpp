// ============================================================================
// test_license_gate.cpp — LicenseGate 授权码校验单测（过渡件）
//
// 契约钉死（docs/plans/2026-11-04-加密验证系统-design.md §2.3/§2.4）：
//   - 期望值以 SHA-256 小写十六进制注入（生产=编译期常量；测试=预生成字面量，
//     本件不 include Qt 头——期望哈希与明码对照表见 design.md §2.4）；
//   - 验证口径：去首尾空白（空格/Tab/CR/LF）→ SHA-256 → 小写十六进制 →
//     与期望值比对（期望值大小写不敏感——构造期归一化）；
//   - 码本身大小写敏感（"Abc-123" ≠ "abc-123"）；
//   - 空码 / 纯空白码 / 空期望 → 恒 false。
// 期望哈希生成命令（管理员 PowerShell，design.md §2.5）：
//   $c=[System.Security.Cryptography.SHA256]::Create()
//   [System.BitConverter]::ToString($c.ComputeHash(
//     [Text.Encoding]::UTF8.GetBytes("码"))).Replace("-","").ToLower()
// ============================================================================

#include <gtest/gtest.h>

#include "modules/08_DeviceManager/LicenseGate.h"

using namespace Scanner::device;

namespace {

// 占位码 JMW-K2-2026 的 SHA-256（正式码人工定后替换——design.md §十二-1）
constexpr char kHashJmwK2[] =
    "7b5d2187bd9146f5e62cc7b0913a1327e0dda31b9f7a3b964aa22223a7c5e980";
// 测试专用码 "Abc-123" 的 SHA-256（大小写敏感断言用）
constexpr char kHashAbc123[] =
    "6c3422a0ed44d5da8c88ffefcc4a7caa24e951bd921257f4c9f594c3fd8e666a";
// 测试专用码 "x" 的 SHA-256（空码拒绝断言用——非空期望配空码）
constexpr char kHashX[] =
    "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881";

} // namespace

TEST(LicenseGate, AcceptsExactCode) {
    LicenseGate g(kHashJmwK2);
    EXPECT_TRUE(g.verify("JMW-K2-2026"));
}

TEST(LicenseGate, TrimsSurroundingWhitespace) {   // 首尾空白（含 CR/LF）不挡正确码
    LicenseGate g(kHashJmwK2);
    EXPECT_TRUE(g.verify("  JMW-K2-2026 \r\n"));
}

TEST(LicenseGate, RejectsWrongCode) {
    LicenseGate g(kHashJmwK2);
    EXPECT_FALSE(g.verify("wrong-code"));
}

TEST(LicenseGate, CaseSensitiveCode) {            // 码含字母时大小写敏感
    LicenseGate g(kHashAbc123);
    EXPECT_TRUE(g.verify("Abc-123"));
    EXPECT_FALSE(g.verify("abc-123"));
}

TEST(LicenseGate, RejectsEmptyAndWhitespaceOnly) {
    LicenseGate g(kHashX);
    EXPECT_FALSE(g.verify(""));                   // 空码
    EXPECT_FALSE(g.verify("   \t  "));            // 纯空白=空码
}

TEST(LicenseGate, AcceptsUppercaseHexExpected) {  // 期望哈希注入为大写也能用（归一化）
    LicenseGate g("7B5D2187BD9146F5E62CC7B0913A1327E0DDA31B9F7A3B964AA22223A7C5E980");
    EXPECT_TRUE(g.verify("JMW-K2-2026"));
}
