#pragma once
// ============================================================================
// cuda_census v3 — CUDA 运行时 API 协作普查（JMW_CENSUS=1 或 enableForTest()）
//
// v2 变更（2026-10-27）：
//  - 补计图族（graphLaunch/beginCap/endCap/graphInst/graphUpload/graphExecUpd）
//    与 mallocHost/hostReg/streamQuery/evElapsed 等——M4 图化后若不计图发射会虚降
//  - 三层口径派生行：COLLAB_C0（全部命令）/C1（剔除 main 行 init malloc+free）/C2（÷帧）
//  - snapshot()/noteFrames()/enableForTest() 进程内 API（gtest 断言用）
// 列数由 StageCount 字段推导（R3B 裁决：字段为唯一事实源）。
//
// v3 变更（2026-10-27 深夜，批 A）：
//  - 方向分类 meta cast 修法（functionParams＝值存储，窄读 kind；旧整读=方向病）
//    ＋cudaMemcpy_ptds(215)/cudaMemcpyAsync_ptsz(225) 补订
//  - 【旁列】mc2D/mca2D（2D 拷贝补计）与 hostAlloc（cudaHostAlloc 补计）
//    三列均**不入 total/C0/C1/C2**（冻结锚，保 A 批「C2 不变」门）
//  - CSV 尾部 COLLAB 行前置 GEN,v3 代际行（按表头名解析，勿按列号）
// ============================================================================

#include <map>
#include <string>

namespace jmw {
namespace census {

// 与内部 StageCount 同构的 POD 快照（供测试/外部读取）
struct Counters {
    long launch = 0, mcH2D = 0, mcD2H = 0, mcD2D = 0;
    long mcaH2D = 0, mcaD2H = 0, mcaD2D = 0;
    // 【v3 旁列】2D 拷贝补计（同步/异步各一，不分方向——stage 行已天然区分），
    // 与 cpp StageRow 同锚点（toCounters 整拷），不入 total（C0/C1/C2 冻结锚）
    long mc2D = 0, mca2D = 0;
    long memset = 0, memsetA = 0, sym = 0, symA = 0;
    long hostReg = 0, hostUnreg = 0, mallocHost = 0, freeHost = 0;
    // 【v3 旁列】cudaHostAlloc 补计——与 cpp StageRow 同锚点（toCounters 整拷），不入 total
    long hostAlloc = 0;
    long mallocAsync = 0, freeAsync = 0;
    long graphLaunch = 0, graphInst = 0, beginCap = 0, endCap = 0;
    long graphUpload = 0, graphExecUpd = 0, hostFunc = 0;
    long streamCreate = 0, streamDestroy = 0, streamQuery = 0;
    long streamSync = 0, devSync = 0;
    long evCreate = 0, evDestroy = 0, evRecord = 0, evQuery = 0, evSync = 0;
    long evElapsed = 0, evWait = 0;
    long malloc = 0, free = 0;
    long total() const {
        return launch + mcH2D + mcD2H + mcD2D + mcaH2D + mcaD2H + mcaD2D +
               memset + memsetA + sym + symA + hostReg + hostUnreg +
               mallocHost + freeHost + mallocAsync + freeAsync +
               graphLaunch + graphInst + beginCap + endCap + graphUpload +
               graphExecUpd + hostFunc + streamCreate + streamDestroy +
               streamQuery + streamSync + devSync + evCreate + evDestroy +
               evRecord + evQuery + evSync + evElapsed + evWait + malloc + free;
    }
};

// 三层派生口径（R1C/R3B 裁决：C1=减 main 行，不改原始行）
struct Derived {
    long c0 = 0;      // 全部阶段 total 之和（原始账本）
    long c1 = 0;      // c0 − main.malloc − main.free（剔除一次性初始化分配）
    double c2 = -1.0; // c1 / frames（稳态每帧；frames<=0 时为 -1）
};

void init();                      // main() 最早调用：env JMW_CENSUS=1 时订阅；未开零开销
void setStage(const char* name);  // 流水线线程入口打阶段标签
bool enableForTest();             // 编程式开启（gtest 用；等效 init 的订阅主体，幂等）
void noteFrames(long n);          // 记录本次运行帧数（derive 用；main 主循环结束处调用）
bool snapshot(std::map<std::string, Counters>& out);  // 进程内账本快照（未启用返回 false）
Derived derive(const std::map<std::string, Counters>& m, long frames);  // 纯函数

}  // namespace census
}  // namespace jmw
