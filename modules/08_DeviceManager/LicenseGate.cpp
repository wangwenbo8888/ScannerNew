// ============================================================================
// LicenseGate.cpp — 授权码校验实现（过渡件）
//
// 哈希用 QtCore QCryptographicHash（08 已链 Qt5::SerialPort，QtCore 同源可用）
// ——不引第三方密码库，固定码比对无需抗碰撞级实现（过渡件，design.md §二）。
// ============================================================================

#include "modules/08_DeviceManager/LicenseGate.h"

#include <QCryptographicHash>
#include <algorithm>
#include <cctype>

namespace Scanner::device {

namespace {

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// 去首尾空白（空格/Tab/CR/LF）——纯空白返回空串
std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

} // namespace

LicenseGate::LicenseGate(std::string expectedSha256Hex)
    : expected_(toLower(trim(std::move(expectedSha256Hex)))) {}

bool LicenseGate::verify(const std::string& code) const {
    const std::string trimmed = trim(code);
    if (trimmed.empty() || expected_.empty()) return false;
    const QByteArray hash = QCryptographicHash::hash(
        QByteArray(trimmed.data(), static_cast<int>(trimmed.size())),
        QCryptographicHash::Sha256);
    // toHex 出来即小写；期望值构造期已归一化小写——直接比对
    return hash.toHex().toStdString() == expected_;
}

} // namespace Scanner::device
