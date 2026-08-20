#pragma once
// ============================================================================
// CrashHandler.h — SEH 崩溃捕获：MiniDump + 标记文件；残留检测（设计方案 §7）
//
//   · install：create_directories + SetUnhandledExceptionFilter；filter 内
//     只做最小安全动作——MiniDumpWriteDump + crash_pending 标记（禁堆分配/
//     锁/日志系统调用/zip——SEH filter 内不安全会二次崩，§7.1 审核 #4）。
//     目录在 install 时解析为绝对路径缓存进 static wchar 缓冲，filter 内
//     零分配；dump 名 = crash_<GetTickCount>.dmp
//   · 打包不在崩溃当场做（§7.2）：下次启动 detectResidualDump 检出残留后，
//     在正常运行环境调 ObsLogger 排障包 API（app T12 接线）
//   · detectResidualDump：扫描 install 缓存目录内 .dmp，返回最旧一份路径
//     （crash_pending 标记存在表示有待打包崩溃）；检出后不删除——打包/
//     删除归调用方
//   · install 幂等：重复调用只更新缓存路径（重设 filter 无害）
// ============================================================================
#include <string>

namespace Scanner::service::crash {

bool install(const std::string& dumpDir);       // SetUnhandledExceptionFilter → dump+标记
bool detectResidualDump(std::string& outPath);  // 检测残留（有则 true + 最旧 .dmp 路径）

} // namespace Scanner::service::crash
