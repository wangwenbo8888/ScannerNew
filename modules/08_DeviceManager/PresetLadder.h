#pragma once
// ============================================================================
// PresetLadder.h — 通用档位梯（261002 按键交互域定稿 §3.3.2「档位梯泛化」）
//
// 通用梯子 Ladder<TStep>：档号 1..N 钳制步进（到顶/底再按无效，不环绕），
// 档值表可整体注入替换（产线定表后 setSteps——档号＝真相源，参数只是投影）。
//
// 四梯一档（三梯一景深）：
//   亮度梯    20 档（每档＝曝光+激光+补光三参组合）——三参唯一合法写入口是
//             换档，禁止绕梯直改（写账归 DeviceManager/ParamStore 既有链）
//   显示远近梯 5 档——显示端摄像机远近预设，只改预览观看不碰采集参数
//   体素密度梯 4 档（菜单①）——点云融合体素哈希密度，换档按新密度重算点云
//   景深档    双态直切（非梯子）：近（默认）↔远——上键双击互换，纯计算侧参数
//
// 纯账本件：只算档位与档值，写账/下发/广播归 DeviceManager。
// 单线程属主：逻辑线程（与 MenuLogic/ParamStore 同口径——本类不加锁）。
// ============================================================================
#include <cstdint>
#include <vector>

namespace Scanner::device {

// ============================================================================
// 通用梯子：钳制步进（不环绕）＋档表整体替换
// ============================================================================
template<typename TStep>
class Ladder {
public:
    Ladder() = default;
    explicit Ladder(std::vector<TStep> steps, int index = -1)
        : steps_(std::move(steps)) {
        // 261004 修：默认中位与 reset() 统一为 (N+1)/2——原算式 N/2+1 在偶数档
        // 时高 1（20 档默认 11 而非定版档 10，N10 帧 B52/L52 ≠ 定版 47/47，
        // 13 个测试断言偏移根因）。261003 定版：亮度梯默认档 10 ≈ 2.9ms/47/47。
        const int mid = (static_cast<int>(steps_.size()) + 1) / 2;   // 默认中位
        if (index < 1) index = mid;
        if (index > static_cast<int>(steps_.size()))
            index = static_cast<int>(steps_.size());
        index_ = steps_.empty() ? 0 : index;
    }

    void setSteps(std::vector<TStep> steps) {          // 整体替换（档数≥1；空忽略）
        if (!steps.empty()) {
            steps_ = std::move(steps);
            if (index_ > static_cast<int>(steps_.size()))
                index_ = static_cast<int>(steps_.size());
        }
    }
    const std::vector<TStep>& steps() const { return steps_; }
    int index() const { return index_; }               // 当前档（1 起；0=空梯）
    void setIndex(int i) {                             // P-5 绝对设档（UI 换档请求；
        if (i >= 1 && i <= static_cast<int>(steps_.size())) index_ = i;   // 越界忽略）
    }

    // 步进 ±1（钳制不环绕）：true=档位变化；false=已到顶/底（无效步，静默）
    bool step(int dir) {
        const int next = index_ + (dir > 0 ? 1 : -1);
        if (next < 1 || next > static_cast<int>(steps_.size())) return false;
        index_ = next;
        return true;
    }
    void reset() { index_ = steps_.empty() ? 0 : (static_cast<int>(steps_.size()) + 1) / 2; }
    // 默认中位档（261003 均分版：档10 ≈ 2.9ms / 47 / 47——启动不灭灯也不刺眼）
    const TStep& current() const { return steps_[static_cast<size_t>(index_ - 1)]; }

private:
    std::vector<TStep> steps_;
    int index_ = 0;
};

// ============================================================================
// ① 亮度梯（三参组合，20 档）
// ============================================================================
struct BrightnessStep {           // 一档＝三参组合（单位同 ParamStore specs）
    double exposureMs = 3.0;      // 曝光 ms（spec 1-100；UI 口径实用 1-5）
    double laserLevel = 40.0;     // 激光亮度 0-100
    double bgLight = 10.0;        // 补光 0-100
};
using LadderStep = BrightnessStep;    // 兼容旧名（260927 G6 期）

/// 缺省均匀曝光表（20 档 min→max 等差）——camera.json 缺 brightnessLadder 节/
/// 非法时兜底（min≤0 / max≤min 回退 0.5~5）
inline std::vector<double> uniformExposureTable(double minMs, double maxMs) {
    if (!(minMs > 0.0) || !(maxMs > minMs)) {
        minMs = 0.5;
        maxMs = 5.0;
    }
    std::vector<double> t;
    t.reserve(20);
    for (int i = 0; i < 20; ++i)
        t.push_back(minMs + static_cast<double>(i) * (maxMs - minMs) / 19.0);
    return t;
}

// 20 档产线基线（261003 口径：激光/补光均分 0~100；261004 曝光档值表化——
// camera.json brightnessLadder.exposureMs 20 元数组逐档注入，缺省均匀 0.5~5）：
//   曝光 = exposureMs[i]（第 i+1 档；逐档自由，非强制等差——方便产线按需定表）
//   补光 0~100 → 步长 100/19 ≈ 5.263
//   激光 0~100 → 步长 100/19 ≈ 5.263
// 非法表（元素数≠20 / 任一 ≤0）回退内置默认。真机定表后亦可 setSteps 整体替换
inline std::vector<BrightnessStep> builtinBrightnessLadder20(
    const std::vector<double>& exposureMs) {
    auto valid = [](const std::vector<double>& t) {
        if (t.size() != 20) return false;
        for (double v : t)
            if (!(v > 0.0)) return false;
        return true;
    };
    std::vector<double> exp = valid(exposureMs) ? exposureMs
                                                : uniformExposureTable(0.5, 5.0);
    std::vector<BrightnessStep> steps;
    steps.reserve(20);
    for (int i = 0; i < 20; ++i) {
        const double t = static_cast<double>(i) / 19.0;   // 0.0 ~ 1.0 均分（激光/补光）
        steps.push_back(BrightnessStep{
            exp[static_cast<size_t>(i)],     // 曝光＝档值表逐档（camera.json 注入）
            t * 100.0,           // 激光 0→100 线性均分
            t * 100.0});         // 补光 0→100 线性均分
    }
    return steps;
}

// 兼容壳：保留 260927 G6 期类名与接口（ladder()/builtinLadder()）
class PresetLadder : public Ladder<BrightnessStep> {
public:
    PresetLadder() : Ladder<BrightnessStep>(builtinLadder()) {}
    static std::vector<BrightnessStep> builtinLadder() {
        return builtinBrightnessLadder20(uniformExposureTable(0.5, 5.0));   // 缺省均匀表
    }
    const std::vector<BrightnessStep>& ladder() const { return steps(); }
};

// ============================================================================
// ② 显示远近梯（5 档）——档值＝显示端远近预设编号（1=最近 … 5=最远；
// 实际摄像机参数映射归 03 显示端消费；档值表产线对账后可 setSteps 替换）
// ============================================================================
class DisplayDistanceLadder : public Ladder<int> {
public:
    // 261004 扩 5→9 档（用户口径：最大档不够大，而每档系数 ×4.72 一按即过头
    // ——范围由档位承担）：中位 5，每档 ×2.36（OSGWidget zoomGears），最大≈×31
    DisplayDistanceLadder() : Ladder<int>({1, 2, 3, 4, 5, 6, 7, 8, 9}) {}
};

// ============================================================================
// ③ 体素密度梯（分辨率设置·菜单①）——261003 产线定版三段步长：
//   0.01~0.1mm 步长 0.01（10 档）→ 0.1~1mm 步长 0.1（9 档）→ 1~5mm 步长 0.5（8 档）
//   共 27 档；档值＝体素边长 mm（越小点云越密越细）
//   07/09 融合累积器按档取密度，换档触发点云重算
// ============================================================================
class VoxelDensityLadder : public Ladder<double> {
public:
    VoxelDensityLadder() {
        std::vector<double> steps;
        for (double v = 0.01; v <= 0.1 + 1e-9; v += 0.01) steps.push_back(v);   // 0.01..0.10
        for (double v = 0.2; v <= 1.0 + 1e-9; v += 0.1) steps.push_back(v);      // 0.2..1.0
        for (double v = 1.5; v <= 5.0 + 1e-9; v += 0.5) steps.push_back(v);     // 1.5..5.0
        setSteps(steps);
        reset();   // index 设中位 14（1.0mm）——不加则 index=0（无效，UI 与梯不一致根因）
    }
};

// ============================================================================
// ④ 景深档（双态直切，非梯子）——近（默认）↔远：两档分辨率完全相同，区别
// 只在计算侧对相应景深区间做屏蔽（激光匹配屏蔽区间归 07/09 对接）；上键双击互换
// ============================================================================
enum class DepthOfField : uint8_t { Near = 0, Far = 1 };

inline DepthOfField toggleDepthOfField(DepthOfField d) {
    return d == DepthOfField::Near ? DepthOfField::Far : DepthOfField::Near;
}

} // namespace Scanner::device
