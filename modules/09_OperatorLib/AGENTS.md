# 09 AGENTS — 算子库

> 先读：本目录 [`09_OperatorLib-说明与交接.md`](09_OperatorLib-说明与交接.md)＋`README.md`（契约速览）；契约全文 [`docs/规范与模板/算子规范.md`](../../docs/规范与模板/算子规范.md)；算子历史说明 `docs/存量档案/算子说明文档/`（存量·终态）；对象速查 [`../../工程地图.md`](../../工程地图.md) §6。

## 一行速记

约 46 个算法算子出 OperatorLib.dll（ns `calib::`）：core/marker（标志点链）＋core/laser＋scanning（激光/融合/GBA）；调度在 07、这里只算。

## 本目录禁令

- **算子三元组契约**（Params/Result/Operator＋pImpl＋末参 `Stream&`）——新算子照《算子规范.md》，旧算子不改名
- **公共头改动＝契约指纹变＝全量重编**（exe 启动校验 09，指纹不一致直接拒绝）；CUDA 五件套命名 `<名>_cuda.h/.cpp/_pimpl.h/_impl.cu`
- 算子无基类无全局态——组合优于继承；`TEST/` 沙箱可自由实验但产物不入库
