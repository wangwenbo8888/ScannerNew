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

// 占位码 88888888 的 SHA-256（正式码人工定后替换——design.md §十二-1）
constexpr char kHash8x8[] =
    "615ed7fb1504b0c724a296d7a69e6c7b2f9ea2c57c1d8206c5afdf392ebdfd25";
// 测试专用码 "Abc-123" 的 SHA-256（大小写敏感断言用）
constexpr char kHashAbc123[] =
    "6c3422a0ed44d5da8c88ffefcc4a7caa24e951bd921257f4c9f594c3fd8e666a";
// 测试专用码 "x" 的 SHA-256（空码拒绝断言用——非空期望配空码）
constexpr char kHashX[] =
    "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881";

} // namespace

TEST(LicenseGate, AcceptsExactCode) {
    LicenseGate g(kHash8x8);
    EXPECT_TRUE(g.verify("88888888"));
}

TEST(LicenseGate, TrimsSurroundingWhitespace) {   // 首尾空白（含 CR/LF）不挡正确码
    LicenseGate g(kHash8x8);
    EXPECT_TRUE(g.verify("  88888888 \r\n"));
}

TEST(LicenseGate, RejectsWrongCode) {
    LicenseGate g(kHash8x8);
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
    LicenseGate g("615ED7FB1504B0C724A296D7A69E6C7B2F9EA2C57C1D8206C5AFDF392EBDFD25");
    EXPECT_TRUE(g.verify("88888888"));
}
