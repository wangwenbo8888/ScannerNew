// ============================================================================
// CrashHandler.cpp — SEH 崩溃捕获实现（设计方案 §7）
//
//   · filter 最小动作清单（§7.1）：GetTickCount 拼文件名（纯栈，无 CRT 格式
//     化/堆）→ CreateFileW → MiniDumpWriteDump → crash_pending 标记（一字节）
//     → CloseHandle → EXCEPTION_EXECUTE_HANDLER。全程零堆分配/零锁/零日志
//   · install 时预解析绝对路径进 static wchar 数组（含尾随分隔符 + 标记
//     完整路径），filter 内不碰 filesystem/CRT 初始化
//   · SEM_NOGPFAULTERRORBOX：免 WER 弹窗挂死无人值守环境（测试子进程/产线）
// ============================================================================
#include "CrashHandler.h"

#include <windows.h>
#include <dbghelp.h>

#include <cwchar>
#include <filesystem>

namespace fs = std::filesystem;

namespace Scanner::service::crash {

namespace {

constexpr size_t kPathCap = 1024;   // 远超 MAX_PATH 的安全余量
constexpr wchar_t kMarkName[] = L"crash_pending";

wchar_t g_dumpDirW[kPathCap];       // 绝对路径 + 尾随 '\'（install 时缓存）
wchar_t g_markPathW[kPathCap];      // crash_pending 标记完整路径
bool g_installed = false;

// --- filter 内追加工具：纯栈写，不触堆/CRT ---

wchar_t* appendStr(wchar_t* p, const wchar_t* s) {
    while (*s) *p++ = *s++;
    return p;
}

wchar_t* appendTick(wchar_t* p, DWORD v) {
    wchar_t digits[12];
    int n = 0;
    do {
        digits[n++] = static_cast<wchar_t>(L'0' + v % 10);
        v /= 10;
    } while (v);
    while (n) *p++ = digits[--n];
    return p;
}

LONG WINAPI crashFilter(EXCEPTION_POINTERS* pep) {
    wchar_t filePath[kPathCap];
    wchar_t* p = appendStr(filePath, g_dumpDirW);
    p = appendStr(p, L"crash_");
    p = appendTick(p, GetTickCount());
    p = appendStr(p, L".dmp");
    *p = L'\0';

    HANDLE hFile = CreateFileW(filePath, GENERIC_WRITE, 0, nullptr,
                               CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile != INVALID_HANDLE_VALUE) {
        MINIDUMP_EXCEPTION_INFORMATION mdei;
        mdei.ThreadId = GetCurrentThreadId();
        mdei.ExceptionPointers = pep;
        mdei.ClientPointers = FALSE;
        MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile,
                          static_cast<MINIDUMP_TYPE>(MiniDumpNormal | MiniDumpWithDataSegs),
                          &mdei, nullptr, nullptr);
        CloseHandle(hFile);
    }

    // crash_pending 标记：一字节，供下次启动残留检测判定"有待打包崩溃"
    HANDLE hMark = CreateFileW(g_markPathW, GENERIC_WRITE, 0, nullptr,
                               CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hMark != INVALID_HANDLE_VALUE) {
        DWORD written = 0;
        WriteFile(hMark, "1", 1, &written, nullptr);
        CloseHandle(hMark);
    }
    return EXCEPTION_EXECUTE_HANDLER;
}

} // namespace

bool install(const std::string& dumpDir) {
    std::error_code ec;
    const fs::path abs = fs::absolute(fs::path(dumpDir), ec);
    if (ec) return false;
    fs::create_directories(abs, ec);
    if (ec && !fs::is_directory(abs)) return false;

    std::wstring dirW = abs.wstring();
    if (dirW.empty() || dirW.back() != L'\\') dirW.push_back(L'\\');
    // 余量：crash_<10 位 tick>.dmp + '\0'
    if (dirW.size() + 20 + 1 > kPathCap) return false;

    for (size_t i = 0; i < dirW.size(); ++i) g_dumpDirW[i] = dirW[i];
    g_dumpDirW[dirW.size()] = L'\0';
    for (size_t i = 0; i < dirW.size(); ++i) g_markPathW[i] = dirW[i];
    for (size_t i = 0; i <= wcslen(kMarkName); ++i)   // 含结尾 L'\0'
        g_markPathW[dirW.size() + i] = kMarkName[i];

    // 免 WER 弹窗（保留进程既有 error mode 其他位）
    SetErrorMode(GetErrorMode() | SEM_NOGPFAULTERRORBOX);

    SetUnhandledExceptionFilter(&crashFilter);
    g_installed = true;
    return true;
}

bool detectResidualDump(std::string& outPath) {
    outPath.clear();
    if (!g_installed) return false;

    // 单一 crash_pending 标记无法逐 dump 区分——按"最旧 .dmp"检出；
    // 标记存在即表示最近一次崩溃未经打包（§7.2，上报归调用方）
    std::error_code ec;
    fs::path oldest;
    fs::file_time_type oldestTime = (fs::file_time_type::max)();   // 括号防 windows.h max 宏

    for (fs::directory_iterator it(g_dumpDirW, ec), end; it != end; it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file(ec)) continue;
        if (it->path().extension() != L".dmp") continue;
        const auto wt = it->last_write_time(ec);
        if (ec) continue;
        if (wt < oldestTime) {
            oldestTime = wt;
            oldest = it->path();
        }
    }
    if (oldest.empty()) return false;
    outPath = oldest.string();
    return true;
}

} // namespace Scanner::service::crash
