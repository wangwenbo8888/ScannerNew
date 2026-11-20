# curve_map_cuda 说明与交接

## 现状

- 来源：2026-11-17 自 factory_calib 整目录镜像（W1 指纹窗批 c200303）；fc 侧 2026-10-08 转正
- 结构：curve_map_cuda.{h,cu} 两件形态（fc 原生，09 五件套惯例豁免已记录）；-fmad=false 专用编译（09 CMakeLists set_source_files_properties）
- 消费方：cmtt_cuda 编排调用；测试经 test_cmtt_cuda 间接覆盖

## 日志

- 261117｜自 fc 引入；-fmad=false 红线入本目录 AGENTS｜人工触发·AI 代笔｜规范 §2.3
