#pragma once
// ============================================================================
// scr_config.h — scan_chain_replay 配置装载（paths / runtime / operators 三节）
// ============================================================================
// 设计：docs/plans/2026-10-04-scan-chain-replay-test-design.md §4
// - paths：数据/标定/CMTT/输出目录；支持 {JMW_SOURCE_ROOT} 占位符（CMake 注入宏）
// - runtime：maxCycles/nominalTemperature/laserFrameParity/warmupCycles/
//   dumpFullFrames/checkCycles/check（断言阈值，两步钉的第二步填）/dedupPath
// - operators：18 算子参数节（逐算子 Params::fromJson 装载；节缺失=默认参数）
// 全容错：文件不存在/节缺失不抛异常，返回 false 或回退默认值。
#include <nlohmann/json.hpp>
#include <string>

namespace scr {

struct ScrPaths {
    std::string dataDir;       // 帧目录（含 left/ right/）
    std::string cameraCalib;   // camera_calib.json
    std::string cmttSidecar;   // curve_map_temp_table.bin（奇号管，兼默认）
    std::string cmttSidecarEven;  // 偶号管 CMTT（可选；空=与奇号管同表）
    std::string outDir;        // 产物输出目录（相对→以 source root 为基）

    bool allExist(std::string& missing) const;
};

struct ScrRuntime {
    int maxCycles = 0;                 // 0=全部；1/2/N 截断（同帧模式：cycle=帧）
    double nominalTemperature = 25.0;  // 名义温（立体/激光温表就近取档）
    int warmupCycles = 1;              // 计时前丢弃的预热 cycle 数
    int dumpFullFrames = 3;            // 前 K cycle 落全量产物，其余仅 summary/timing
    int checkCycles = 10;              // gtest 断言默认截断
    nlohmann::json check = nlohmann::json::object();  // 断言阈值（空=report-only）
    // 去重路径：operator=算子内 GPU 去重（默认，dedupMode 照 steger_fast 节）；
    // cpu=外挂 CPU 位移式去重【对照 oracle】（harness 强制 dedupMode=0 防双重去重）
    std::string dedupPath = "operator";
    // 激光中心提取算子：steger=现役（默认）｜flat=flat_center 平顶专用（无去重
    // 步骤，与 dedupPath=cpu 互斥——fromJson 硬校验）。步骤槽位键恒为 "steger_fast"。
    std::string extractor = "steger";
    // 【150FPS·方案 A】CCL 路径：gpu=原 RegionAnalyzerCUDA（含三次全设备同步，
    // 串行参与 6.8ms/帧）；cpu=RLE 并查集单线程版（掩膜下载后纯 CPU，0.4-1.2ms，
    // 序＝光栅首遇升序，与 GPU 版对拍验证）。流水线模式强制 cpu。
    std::string cclPath = "cpu";
    // 只跑标记点链：跳过激光链与配准（CLI --markers-only 置位；不出链级产物，
    // 仅逐 cycle 落标记点圆心坐标＋法线 CSV）
    bool markerOnly = false;
    // —— frame_fuse 兜底链内验收门（2026-10-15 参数化，原为 harness 硬编码；
    //    数据定标：真值残差 ≤0.2mm → rmse 门 2.0mm＝10 倍余量；垃圾解 78~169mm
    //    全拒；对称错解 <1mm 由算子签名法线＋并列弃权扛）——
    int fuseMinMatched = 4;      // 配对数下限（真值分布 3~7；4 兼顾稀疏帧）
    double fuseMaxRmseMm = 2.0;  // rmse 上限（mm）
    // 兜底子集装配上限（2026-10-15 批1 P0：与算子 triangleMaxPoints 解绑——
    // 调大算子上限想"大库直配"时不再连带放大子集；两侧取小后 clamp [6,64]）
    int fuseSubsetMax = 16;
    // 断链重锁：【2026-10-15 人工裁决取消并删码】——A/B 对照实验（同代码仅切开关，
    // ALL 399 帧 10mm 门）成链率完全相同（318 vs 318，0.0pp），判据通过。
    // 历史与依据见 tests/registration_optimize/重锁取消-决策与执行方案.md。
    // 新点发号查重半径（mm，全局系）：光流/重锁路径发新号前先在地图里找最近条目，
    // 命中半径内沿用旧号——防同点重复领号永久污染地图（2026-10-15 裁决，对齐兜底
    // 路径做法）。独立于 matchDistThresh（查重量的是累积漂移，不是帧间运动）。
    double mintDedupRadiusMm = 5.0;
    // 误差趋势门（×开局基线中位，2026-10-15 加固）：光流成功帧误差若超"前 50 个
    // 成功帧误差中位 × 本倍率"判失败走降级链——专抓温漂/斜视角等"全体一起慢慢
    // 变坏"（帧间差分防线结构性看不见）。0=关。校准（ALL 399 帧实测）：健康数据
    // 极值=基线 7.3 倍，温漂场景≈43 倍→10 倍两侧各留余量。
    double flowRmseDriftRatio = 10.0;
};

// 【五链重构】fivechain 节（executor=fivechain 时消费；字段总表见实现规范附录 7.3）
struct ScrFiveChain {
    std::string executor = "serial";     // serial|pipe|fivechain（模板默认 serial 保 gtest 稳定）
    int graySlots = 8;                   // 显存灰度环槽数
    int rleSlots = 8;                    // RLE 双环槽数（=A 池容量）
    int rleCap = 65536;                  // 单侧行程容量上限
    int rleWindow = 4096;                // 【E7】RLE 按实拷常态窗口（行程条数；钳 [64, rleCap]，实测行程 ~1.1k/帧＝3.7× 余量；调小＝每帧走补拷＝欠预测压测臂）
    int markerLanes = 3;                 // CPU链1 实例数（P2 固定 1）
    int bCap = 8, cCap = 8;              // B/C 通道容量
    int dQueueCap = 8, dPoolSlots = 4;   // D 队列/池
    int eQueueCap = 8, ePoolBlocks = 9, eMaxPoints = 34237;   // E 队列/池/满帧点数
    int fCap = 8, trCap = 8;             // F/TR 容量
    int maxInflight = 0;                 // 重排/寄存 map 断言界（0=auto=Σ容量+在制推导）
    int memCapMB = 512;                  // fivechain 结构内存硬上限
    long long voxelReserve = 12000000;   // 体素表启动槽位。⚠️2026-10-23 勘误+修正：旧值 8M 注释「5.1M×1.25→8M 负载64%零扩容」系误读——实测 ALL 399 帧终值 ~5.7M 唯一体素，恰在 ~371 帧跨 8M×70%=5.6M 阈值触发同步 rehash（单帧尖峰 22-26ms，诊断实锤：cudaMalloc~400MB＋搬移落在 wallFuse 计时窗）；12M（70% 线 8.4M ≫ 5.7M）全程零触发，实测 max wallFuse 25.4→7.5ms。换更大数据集须按「终值体素 ÷ 0.7」重估
    double voxelSizeMm = 0.25;           // 体素尺寸（产线现口径）
    double voxelLoadFactor = 0.75;       // 负载断言线（超限 fail-fast；算子 70% 自动扩容在前）
    long long voxelGrowSlots = 2000000;  // 扩容增量（收敛防 memCap 击穿）
    std::string pinRoles = "1a,1b,marker,ch2";  // 绑核角色集（E1：ch2=CPU链2 绑核2-LP1）
    bool priorityBoost = true;           // 绑核线程提最高优先级
    // CUDA链1 二分（1a/1b 双流交叉）：默认开——2026-10-21 调优会话 A/B/A/B
    // 实测反超单线程（131.8-137.2 vs 113.5-118.7 FPS）＋ALL 399 对拍 0 diffs；
    // 历史 42.8 FPS 病理未复现，诱因模式（自旋×最高优先级）已改 cv 阻塞＋
    // lmask 多槽环。env JMW_FCSPLIT=0 强制回退单线程（病理环境 kill-switch）
    bool splitChain1 = true;
    int lmaskSlots = 4;                  // lmask D2D 环槽数（2-4）。默认 4（2026-10-23 ABBA 两对同向 −0.1~0.2ms——它是全流水线在飞距最紧的闸，其余环均 8；VRAM +~9.4MB）
    int holdSkip = 0;                    // >0=全部帧按 hold 处理（零上传路径演练）
    int slowChainMs = 0;                 // 拖慢注入（S1-S4）
    std::string failInject;              // 失败注入（F1-F4；空=off）
    int kRleCapOverride = 0;             // R1 溢出注入
    int eBlockSizePct = 100;             // R2 截断注入
    int regWakeP99UsMax = 1000;          // 链3 唤醒 P99 阈值（μs）
    int dbgInterval = 32;                // JMW_FCDBG 实时行间隔（帧）
    double pushBudgetMs = 0.8;           // 链1 生产者侧 push 合计预算（P5 判定门）
};

struct ScrConfig {
    ScrPaths paths;
    ScrRuntime runtime;
    ScrFiveChain fivechain;
    nlohmann::json operators = nlohmann::json::object();  // name → 参数节

    /// 展开路径中的 {JMW_SOURCE_ROOT} 占位符；outDir 相对路径以 source root 为基
    static std::string expandPath(const std::string& raw);

    /// 从 JSON 文件装载（缺节用默认；失败返 false 填 err，不抛异常）
    static bool fromFile(const std::string& path, ScrConfig& out, std::string& err);

    /// 从 JSON 对象装载（测试用）
    static bool fromJson(const nlohmann::json& j, ScrConfig& out, std::string& err);

    /// 取算子参数节（缺该键→空 json 对象=算子默认参数）
    nlohmann::json opParams(const std::string& opName) const;

    /// 默认配置文件查找：--config 参数 > exe 旁 config.json > SCR_SOURCE_DIR/config.json
    static std::string defaultConfigPath();
};

} // namespace scr
