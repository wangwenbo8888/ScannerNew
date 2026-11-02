> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# laser_cloud_fuse · 激光点云融合（CPU）

## 1. 目标

- 功能：激光点云 CPU 体素融合（每体素记首个落入点），兼作后处理法线的宿主
- 方法/参数：体素量化＋SoA 平坦哈希；voxelSize=0.5mm、saturationThreshold=5；产线实时融合用 CUDA 版
- 注意点：有状态非线程安全；FusedPointPtr 勿跨帧持有（重分配后失效）

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- CPU 体素融合（每体素记首个落入点）：达成｜主类 `LaserCloudFuseCPU`（laser_cloud_fuse_cpu.h）：21bit/轴打包键→splitmix64＋tag 预过滤＋线性探测的 SoA 平坦哈希；每体素记首点＋计数、超 threshold 丢弃（threshold=1 即纯去重）
- 方法参数 voxelSize=0.5mm、saturationThreshold=5：达成｜代码实测＝卡默认，reserveVoxelCount=65536 槽、负载 >0.7 自动 2 倍扩容 rehash；预设 fine（0.2mm/8 点/256K 槽）/coarse（1.0mm/3 点/16K 槽）
- 兼作后处理法线的宿主：达成｜07 postprocess/NormalStage.cpp 临时实例化作 `LaserCloudNormalCPU` 宿主；`GatherVoxelNeighbors` O(1) 供法线邻域查询
- 产线实时融合让位 CUDA 版：达成｜产线实时融合用 laser_cloud_fuse_cuda，本件仅后处理路径在用
- 性能/质量锚：达成｜~8M 点/秒、40K 点 avg<5ms/p99<10ms、经 100K 帧 40 亿点压测；保留率 ≥0.5 Normal／0.1~0.5 Degraded／<0.1 Warning
- 约束（有状态非线程安全／FusedPointPtr 勿跨帧持有）：达成（已固化）｜坐标每轴 ±104.8 万体素超范围抛 overflow_error；陷阱已记：指针下次 Execute 重分配后失效、SetParams 改 reserveVoxelCount 须手动 Reserve()
- 独立验证：部分达成｜tests/test_laser_cloud_fuse_cpu.cpp 在，但本目录验证未独立挂账——随 04 后处理链验证

## 3. 问题与方法

- 〔待定〕问题：本目录验证未独立开展（无独立挂账）｜方法：随 04 后处理链验证｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制｜规范 §2.3
- 261102｜转四段新制：目标待人定稿；问题段自下一步+全局挂账迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段按存量参考卡颗粒度扩充（源：docs/存量档案/算子说明文档＋代码实测校准）｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
