#pragma once
// ============================================================================
// ParamStore.h — 参数账本（协议 260831：无 ACK——Done=写成败直通）
// 设备参数（曝光/补光/激光/分辨率等）全工程唯一真相源：UI 纯视图双向同步、
// 按键/滑条两入口同队列串行、下发失败不改账、人工触发才落盘。
// 写成功→改账→广播；写失败→保旧值+弹回事件。开机：有档读档无档统一初始值
// （存取经注入——06 无直链）。
//
// 口径钉死（批2 去 confirmed）：
//   - Dispatch 完成回调 void(bool ok)：ok=写成败（CommandChannel send 的
//     onDone 同步回告直通——写队列即返，无 ACK 确认语义）；
//   - 在途后值胜出：同 key 再 setValue 覆盖在途回调（gen 判据），旧回调到达
//     无论成败不改账不弹回；bootstrap 清在途（会话重启语义，迟到回调同废）；
//   - 入口即钳：setValue/setEntryDirect/bootstrap 读档均钳回 spec 范围；
//   - 序列化手写极简格式 "key=value;..."（08 不引 json 库；06 只是仓库不管
//     格式）；source 不入档（读档恒 source=Boot）。
//
// 单线程属主：逻辑线程（与 MenuLogic 同口径）——本类不加锁；Dispatch 完成回
// 调须回逻辑线程再调（DeviceManager 负责切线程）。
// ============================================================================
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace Scanner::device {

struct ParamSpec {             // 参数字段定义（08 归口——「字段定义归 08」红线）
    std::string key;
    double def = 0.0;
    double min = 0.0, max = 0.0;
};

struct ParamEntry {
    double value = 0.0;
    enum class Source { Boot, Ui, Key } source = Source::Boot;
};

class ParamStore {
public:
    // 完成回调：ok=写成败（盲发直通——写队列即返）
    using Done = std::function<void(bool ok)>;
    // 下发回调（DeviceManager 接 CommandChannel）：key/期望值/完成回调
    using Dispatch = std::function<void(const std::string&, double, Done)>;
    // 存取回调（06 注入——不带脑子的仓库）
    using Persist = std::function<bool(const std::string& json)>;     // 落盘
    using Load = std::function<std::string()>;                        // 读档（空串=无档）

    ParamStore(std::vector<ParamSpec> specs, Dispatch dispatch);

    // 开机装载：Load 有档→按档改账（source=Boot）；无档→默认值。档内未知
    // key/坏数值段忽略，越界钳回范围。逐参数广播 onParamChanged（含默认值项）。
    // 在途清空（会话重启语义）。
    void bootstrap(Load load);
    // 设值（UI 滑条/按键步进两入口同此）：入口即钳→排队下发——ok 才改账
    // （改前查在途：同 key 在途新值直接覆盖旧在途回调失效——后值胜出）；
    // 均广播 onParamChanged（改账后）。未登记 key 不下发不广播。
    void setValue(const std::string& key, double v, ParamEntry::Source src);
    // 落盘（「保存为本工程参数」人工触发才调）："key=value;..." 全账串行化
    bool persist(Persist p) const;
    // 弹回事件（写失败——UI 把控件弹回旧值）
    std::function<void(const std::string& key, double oldValue)> onReject;
    // 改账广播（UI 同步显示——含 bootstrap 载入）
    std::function<void(const std::string& key, const ParamEntry&)> onParamChanged;

    ParamEntry get(const std::string& key) const;   // 未登记 key → value=0/Boot
    bool has(const std::string& key) const;
    bool pending(const std::string& key) const;     // 该 key 有在途未决下发？

    // 测试/特殊路径：直接改账不经下发、不广播（bootstrap 同源静默写；越界同钳）
    void setEntryDirect(const std::string& key, double v, ParamEntry::Source);

private:
    struct InFlight { double expect; uint64_t gen; };   // 在途（gen=后值胜出判据）
    std::map<std::string, ParamSpec> specs_;
    std::map<std::string, ParamEntry> entries_;
    std::map<std::string, InFlight> inflight_;
    Dispatch dispatch_;
    uint64_t genSeq_ = 0;
};

} // namespace Scanner::device
