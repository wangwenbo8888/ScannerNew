// census 探针断言化测试（M0 验证基建）：进程内 snapshot/derive 链路自证
#include "cuda_census.h"

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <map>

// ── 用例1：稳态乒乓分配必须被抓到（P0-c 的回归锚）──
TEST(Census, MallocPingPongSteadyState) {
    if (!jmw::census::enableForTest()) GTEST_SKIP() << "CUPTI unavailable";
    jmw::census::setStage("t_init");
    void* p = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&p, 1 << 20));
    ASSERT_EQ(cudaSuccess, cudaFree(p));  // 初始化期 1 对

    jmw::census::setStage("t_loop");
    constexpr int N = 32;
    for (int i = 0; i < N; ++i) {  // 乒乓 = 每帧 malloc+free
        ASSERT_EQ(cudaSuccess, cudaMalloc(&p, 1 << 20));
        ASSERT_EQ(cudaSuccess, cudaFree(p));
    }
    jmw::census::noteFrames(N);
    std::map<std::string, jmw::census::Counters> s;
    ASSERT_TRUE(jmw::census::snapshot(s));
    ASSERT_GE(s.count("t_loop"), 1u);
    EXPECT_EQ(s.at("t_loop").malloc, N);  // 乒乓被抓到
    EXPECT_EQ(s.at("t_loop").free, N);
    auto d = jmw::census::derive(s, N);
    EXPECT_GT(d.c0, 0);
    EXPECT_EQ(d.c1, d.c0);  // 本测无 main 行 → c1==c0（warn 路径）
    EXPECT_DOUBLE_EQ(d.c2, (double)d.c1 / N);
}

// ── 用例5：memset 落列裁决（52.07 图臂 fuse memset=150/memsetA=0 悬案单测定案）──
// 判读矩阵（批A T0）：
//   全过 → census 无缺陷，图臂现象为当晚状态，不改 case
//   5a 挂 → 编译期/运行时 CUPTI 错位（概率极低）→ 核对 cupti64 DLL 同源
//   5a 过 5b 挂 → 默认流走 ptsz(235) → c1 落地 fallthrough＋订阅
TEST(Census, MemsetColumnAdjudication) {
    if (!jmw::census::enableForTest()) GTEST_SKIP() << "CUPTI unavailable";
    jmw::census::setStage("t_memset5");
    void* dummy = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dummy, 16));
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&s));

    // 5a：显式流 50 笔 async → 必须全落 memsetA
    for (int i = 0; i < 50; ++i) cudaMemsetAsync(dummy, 0, 4, s);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(s));
    std::map<std::string, jmw::census::Counters> sa;
    ASSERT_TRUE(jmw::census::snapshot(sa));
    EXPECT_EQ(50, sa.at("t_memset5").memsetA) << "5a: explicit-stream memsetAsync";
    EXPECT_EQ(0, sa.at("t_memset5").memset) << "5a: sync column must stay 0";

    // 5b：默认流（stream=0）50 笔 async → 仍应落 memsetA（ptsz 变体探针）
    for (int i = 0; i < 50; ++i) cudaMemsetAsync(dummy, 0, 4, 0);
    std::map<std::string, jmw::census::Counters> sb;
    ASSERT_TRUE(jmw::census::snapshot(sb));
    EXPECT_EQ(100, sb.at("t_memset5").memsetA) << "5b: default-stream memsetAsync (ptsz probe)";
    EXPECT_EQ(0, sb.at("t_memset5").memset) << "5b: sync column must stay 0";

    // 5c：同步版 50 笔对照 → 落 memset
    for (int i = 0; i < 50; ++i) cudaMemset(dummy, 0, 4);
    std::map<std::string, jmw::census::Counters> sc;
    ASSERT_TRUE(jmw::census::snapshot(sc));
    EXPECT_EQ(50, sc.at("t_memset5").memset) << "5c: sync memset";
    EXPECT_EQ(100, sc.at("t_memset5").memsetA) << "5c: async column unchanged";

    cudaStreamDestroy(s);
    cudaFree(dummy);
}

// ── 用例3：方向分类回归（kindfix 哨兵，修前必挂）──
// 旧病：reinterpret_cast<intptr_t>(pr[3]) 整读 8B＝kind+padding 垃圾 → D2H 恒落
// default（mcaH2D==200/mcaD2H==0 必挂）。同时是 functionParams 值存储语义哨兵。
TEST(Census, MemcpyDirectionClassification) {
    if (!jmw::census::enableForTest()) GTEST_SKIP() << "CUPTI unavailable";
    jmw::census::setStage("t_dir3");
    unsigned char host[256] = {0};
    void* dev = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dev, 256));
    for (int i = 0; i < 100; ++i)
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev, host, 64, cudaMemcpyHostToDevice));
    for (int i = 0; i < 100; ++i)
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(host, dev, 64, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 100; ++i)
        ASSERT_EQ(cudaSuccess, cudaMemcpy(dev, host, 64, cudaMemcpyHostToDevice));
    for (int i = 0; i < 100; ++i)
        ASSERT_EQ(cudaSuccess, cudaMemcpy(host, dev, 64, cudaMemcpyDeviceToHost));
    std::map<std::string, jmw::census::Counters> s;
    ASSERT_TRUE(jmw::census::snapshot(s));
    const auto& c = s.at("t_dir3");
    EXPECT_EQ(100, c.mcaH2D) << "async H2D (修前 padding 病时 D2H 混入=200)";
    EXPECT_EQ(100, c.mcaD2H) << "async D2H (修前必挂=0)";
    EXPECT_EQ(0, c.mcaD2D);
    EXPECT_EQ(100, c.mcH2D) << "sync H2D";
    EXPECT_EQ(100, c.mcD2H) << "sync D2H (修前必挂=0)";
    cudaFree(dev);
}

// ── 用例6：cudaHostAlloc 旁列补计（v3 旁列设计：计入 hostAlloc 列、不入 total）──
TEST(Census, HostAllocSideColumn) {
    if (!jmw::census::enableForTest()) GTEST_SKIP() << "CUPTI unavailable";
    jmw::census::setStage("t_halloc6");
    std::map<std::string, jmw::census::Counters> s0;
    ASSERT_TRUE(jmw::census::snapshot(s0));
    const long c0_before = jmw::census::derive(s0, 0).c0;
    void* p = nullptr;
    ASSERT_EQ(cudaSuccess, cudaHostAlloc(&p, 4096, cudaHostAllocDefault));
    std::map<std::string, jmw::census::Counters> s;
    ASSERT_TRUE(jmw::census::snapshot(s));
    const auto& c = s.at("t_halloc6");
    EXPECT_EQ(1, c.hostAlloc) << "cudaHostAlloc 补计入列";
    EXPECT_EQ(0, c.mallocHost) << "旁列设计：不得混入 mallocHost（in-total）";
    // 冻结锚：旁列不入 total → derive C0 增量必须为 0
    EXPECT_EQ(c0_before, jmw::census::derive(s, 0).c0) << "hostAlloc 不得入 C0";
    ASSERT_EQ(cudaSuccess, cudaFreeHost(p));
    std::map<std::string, jmw::census::Counters> s2;
    ASSERT_TRUE(jmw::census::snapshot(s2));
    EXPECT_EQ(1, s2.at("t_halloc6").freeHost);
}

// ── 用例4：2D 旁列冻结锁（v3 冻结锚：mc2D/mca2D 计数不入 C0/C1/C2）──
// 设计注：宿主侧用 pinned（cudaHostAlloc 亦为旁列，不扰 C0）；分页内存会被运行时
// 内部 staging 引入杂散 API 计数。快照取于 streamSync 之前（sync 本身计 streamSync 列）。
TEST(Census, Memcpy2DSideColumnFreezeLock) {
    if (!jmw::census::enableForTest()) GTEST_SKIP() << "CUPTI unavailable";
    jmw::census::setStage("t_2d4");
    unsigned char* host = nullptr;
    ASSERT_EQ(cudaSuccess, cudaHostAlloc(&host, 256, cudaHostAllocDefault));
    void* dev = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dev, 256));
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&s));
    std::map<std::string, jmw::census::Counters> s0;
    ASSERT_TRUE(jmw::census::snapshot(s0));
    const long c0_before = jmw::census::derive(s0, 0).c0;
    const long mcaH2D_before = s0.at("t_2d4").mcaH2D;

    for (int i = 0; i < 50; ++i)
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(dev, 64, host, 64, 64, 4, cudaMemcpyHostToDevice));
    for (int i = 0; i < 50; ++i)
        ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(dev, 64, host, 64, 64, 4, cudaMemcpyHostToDevice, s));
    // API ENTER 回调在调用时刻已入账——先快照后收尾（sync 计入 streamSync 列不扰断言）
    std::map<std::string, jmw::census::Counters> snap;
    ASSERT_TRUE(jmw::census::snapshot(snap));
    const auto& c = snap.at("t_2d4");
    EXPECT_EQ(50, c.mc2D) << "sync 2D 补计入旁列";
    EXPECT_EQ(50, c.mca2D) << "async 2D 补计入旁列";
    EXPECT_EQ(mcaH2D_before, c.mcaH2D) << "2D 不得混入 1D 方向列";
    EXPECT_EQ(c0_before, jmw::census::derive(snap, 0).c0) << "冻结锚：旁列不入 C0";
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(s));
    cudaStreamDestroy(s);
    cudaFree(dev);
    cudaFreeHost(host);
}

// ── 用例2：图发射必须被计数（M4 防虚降核心）──
TEST(Census, GraphLaunchCounted) {
    if (!jmw::census::enableForTest()) GTEST_SKIP() << "CUPTI unavailable";
    jmw::census::setStage("t_graph");
    void* dummy = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dummy, 16));
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&s));
    // 捕获一段三内核序列
    ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < 3; ++i) cudaMemsetAsync(dummy, 0, 4, s);
    cudaGraph_t g = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(s, &g));
    cudaGraphExec_t ge = nullptr;
    ASSERT_EQ(cudaSuccess, cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    // 捕获期 3 笔 memsetAsync 是真实 API 调用（被录入图）——先记账
    std::map<std::string, jmw::census::Counters> before;
    ASSERT_TRUE(jmw::census::snapshot(before));
    const long memsetDuringCapture = before.at("t_graph").memsetA;
    ASSERT_EQ(cudaSuccess, cudaGraphLaunch(ge, s));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(s));

    std::map<std::string, jmw::census::Counters> snap;
    ASSERT_TRUE(jmw::census::snapshot(snap));
    const auto& c = snap.at("t_graph");
    EXPECT_EQ(c.graphLaunch, 1);   // 图发射计 1 笔
    EXPECT_GE(c.beginCap, 1);      // 捕获动作入账
    EXPECT_GE(c.endCap, 1);
    EXPECT_GE(c.graphInst, 1);     // 实例化入设置账（稳态应为 0——由 C1/C2 分层判读）
    // 图回放不产生新的 memset API（节点≠API）：仅捕获期那 3 笔
    EXPECT_EQ(c.memsetA, memsetDuringCapture);
    EXPECT_EQ(memsetDuringCapture, 3);
    EXPECT_GE(c.streamCreate, 1);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaStreamDestroy(s);
    cudaFree(dummy);
}
