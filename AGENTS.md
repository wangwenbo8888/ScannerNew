# AGENTS.md — AI 入口（只做路由，不存内容）

> **先读三份**：①[`工程地图.md`](工程地图.md)（§0 全局地图＋文档在哪＋§6 对象速查）→ ②[`开发进度.md`](开发进度.md)（限读：快照＋关键决策＋近 10 条日志＋交接·挂账）→ ③[`工程全景图.md`](工程全景图.md)（分层架构＋模块速查 §5）。
> 文档怎么建、放哪、怎么登记：[`工程文档管理规范.md`](工程文档管理规范.md)。

## 目录一行一棵

- `base/` 共享内核（types.h＋EventBus，禁业务代码）
- `modules/` 11 业务模块（01 标定 ~ 11 部署；09＝算子库 OperatorLib）
- `app/` 应用入口与装配（LeadScannerK2.exe；AppContext 组合根）
- `cmake/` `config/` `scripts/` 构建辅助 / 随软件运行时配置 / 交付脚本
- `docs/` 全部文档（plans / 流水线 / 开发需要的信息 / 规范与模板 / 用户文档＋存量终态五区——见地图 §2）
- `factory_calib/` 厂家标定独立子工程（AI 只读禁区，见其 AGENTS.md）
- `build*/` 构建产物，不入库

对象职责与文件路径查《工程地图.md》§6；模块/算子目录详情看该目录两件套（说明与交接＋AGENTS，落地登记见地图 §3）。

## 红线速记（全文见《工程文档管理规范.md》§4）

- `factory_calib/` **AI 只读保护区**——不读不改不编不测
- `build*/` 不入库；`base/` 不放业务代码；根目录文档只 6 份不再新增
- 删文件/退役必须人批准；封存（存量·终态）目录只出不进
- AI 发现现状与规范不符：**只报告、不动手**

## 构建速查（日常优先 Release）

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure   # 产物在 build/bin/Release
```

VS2022：开 `build/LeadScannerK2.sln` → 启动项 LeadScannerK2 → Release → F5（调试工作目录已指 bin）。
依赖路径与构建细节：《环境配置汇总.md》；真机/联调现状与易踩坑：《开发进度.md》快照。
