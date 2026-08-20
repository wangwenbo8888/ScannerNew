#pragma once
// ============================================================================
// DefaultCommands.h — 首批命令目录注册表（P1-T6）
// 设计基准：docs/plans/2026-08-20-可观测性模块10设计方案.md §3.2 命令目录表
//   · handler/pre 全空——接线方（P1-T14–T16）装配时填入
//   · 实施期两处补正（注释即文档）：
//     ① system_ready（app 自检全过 S1→S2）不是工作流命令，无「收尾」概念
//       → 收尾禁用型（startedEvent≠0 && finishedEvent=kNoEvent）；registerCommand
//         校验已放宽放行（notifyCompleted 对其 fail，见 CommandGate.cpp）
//     ② finish_calibration/finish_postprocess 的 gateOp 用三态门禁键
//       "calibrating"/"postprocessing"（仅 S3/S6 放行，StateMachine::canOperate
//       同步扩充）——"calibrate"/"postprocess" 仅 S2 放行，收尾命令在 S3/S6 态
//       下会被误拒
//   · 05 查询口登记（P5-T16，设计 §9 05 行）：05 编辑不走 gate 不占态
//     （S2 内交互会话）——无命令注册、不占上方 7 命令目录；进入前置经
//     canOperate("edit")（StateMachine 已有键，仅 S2 放行）+ 业务谓词
//     『可编辑点云存在』由 05/app 侧自查（谓词注入同 02-① pre 模式，
//     查 06 PointCloudBuffer）——05 落地时用
// ============================================================================
#include "CommandGate.h"
#include <vector>

namespace Scanner::service {

// 首批 7 命令；返回值按值（RVO/移动），调用方逐条 registerCommand
inline std::vector<CommandGate::Spec> makeDefaultCommandSpecs() {
    std::vector<CommandGate::Spec> specs;

    CommandGate::Spec systemReady;
    systemReady.name = "system_ready";                  // app 自检全过 S1→S2
    systemReady.gateOp = "";                            // S1 无前置态可门禁（内部命令）
    systemReady.startedEvent = EventType::SystemReady;
    systemReady.finishedEvent = CommandGate::kNoEvent;  // 收尾禁用型（补正①）
    specs.push_back(std::move(systemReady));

    CommandGate::Spec startCalib;
    startCalib.name = "start_calibration";              // S2→S3；01-①
    startCalib.gateOp = "calibrate";
    startCalib.startedEvent = EventType::CalibStarted;
    startCalib.finishedEvent = EventType::CalibFinished;
    specs.push_back(std::move(startCalib));

    CommandGate::Spec startScan;
    startScan.name = "start_scan";                      // S2→S4/S5；02-①
    startScan.gateOp = "scan";                          // payload=ScanMode 0/1，状态机矩阵判
    startScan.startedEvent = EventType::ScanStarted;
    startScan.finishedEvent = EventType::ScanStopped;
    specs.push_back(std::move(startScan));

    CommandGate::Spec finishScan;
    finishScan.name = "finish_scan";                    // 触发型：S4/S5 内点火 GBA
    finishScan.gateOp = "scanning";                     // 仅 S4/S5 放行
    finishScan.startedEvent = CommandGate::kNoEvent;    // submit 只点火不切态
    finishScan.finishedEvent = EventType::ScanStopped;  // 切态在 ⑩ notifyCompleted→S2
    specs.push_back(std::move(finishScan));

    CommandGate::Spec startPost;
    startPost.name = "start_postprocess";               // S2→S6；04
    startPost.gateOp = "postprocess";
    startPost.startedEvent = EventType::PostProcessStarted;
    startPost.finishedEvent = EventType::PostProcessFinished;
    specs.push_back(std::move(startPost));

    CommandGate::Spec finishPost;
    finishPost.name = "finish_postprocess";             // S6 内点火；收尾 S6→S2
    finishPost.gateOp = "postprocessing";               // 三态门禁键：仅 S6 放行（补正②）
    finishPost.startedEvent = CommandGate::kNoEvent;
    finishPost.finishedEvent = EventType::PostProcessFinished;
    specs.push_back(std::move(finishPost));

    CommandGate::Spec finishCalib;
    finishCalib.name = "finish_calibration";            // 触发型：S3 内触发收尾流程
    finishCalib.gateOp = "calibrating";                 // 三态门禁键：仅 S3 放行（补正②）
    finishCalib.startedEvent = CommandGate::kNoEvent;
    finishCalib.finishedEvent = EventType::CalibFinished;
    specs.push_back(std::move(finishCalib));

    return specs;
}

} // namespace Scanner::service
