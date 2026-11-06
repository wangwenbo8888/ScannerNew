#pragma once
// ============================================================================
// LicenseGate.h — 授权码校验（过渡件）
//
// 真狗（Virbox）到货前以「固定验证码」顶位加密狗自检项（license）：
// main.cpp 启动弹窗收码 → AppContext::submitLicenseCode → 本件比对 →
// notifySelfCheckItem("license", ok)——失败走既有软门禁（滞留 S1）。
// 终态演进：本件由 IDongle/DongleGuard（Virbox SDK 封装）接替，
// 见 docs/plans/2026-11-04-加密验证系统-design.md §二/§七。
//
// 期望值以 SHA-256 十六进制注入：生产=编译期常量（人工定码后替换，
// 换码命令见 design.md §2.5）；测试=任意注入（预生成字面量）。
// 无 UI 依赖、无状态——08 模块「许可证」家族（CMakeLists 首行注记）。
// ============================================================================

#include <string>

namespace Scanner::device {

class LicenseGate {
public:
    // expectedSha256Hex：期望码的 SHA-256 十六进制串（32B=64 字符；
    // 大小写不敏感——构造期归一化为小写）
    explicit LicenseGate(std::string expectedSha256Hex);

    // 校验口径：去首尾空白（空格/Tab/CR/LF）→ SHA-256 → 小写十六进制
    // → 与期望值比对。码本身大小写敏感。空码/空期望恒 false。
    bool verify(const std::string& code) const;

private:
    std::string expected_;   // 归一化小写十六进制
};

} // namespace Scanner::device
