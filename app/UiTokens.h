#pragma once
// ============================================================================
// UiTokens.h — 排版档位唯一事实源（三轮审核定稿 261117）
// 口径：首轮只做高度归并＋token 化；间距六档为二轮目标不设常量；
//       图标同值透传。回迁对位：app/UiTokens.h 与 shell/UiTokens.h 逐字节一致。
// ============================================================================
namespace UiT {
// 字号七档（px）
constexpr int FontAux        = 12;  // 辅助信息
constexpr int FontBody       = 14;  // 正文
constexpr int FontTitle      = 16;  // 标题
constexpr int FontBlock      = 18;  // 区块强调
constexpr int FontOverlay    = 20;  // 远距正文·overlay
constexpr int FontBannerNear = 26;  // 横幅近距
constexpr int FontBannerFar  = 40;  // 横幅远距（默认案待 A/B 裁定）
// 图标五档（px，同值透传）
constexpr int IconXs = 11; constexpr int IconSm = 14; constexpr int IconMd = 18;
constexpr int IconLg = 20; constexpr int IconXl = 28;
// 控件高三档（px）
constexpr int CtlSm = 28; constexpr int CtlMd = 36; constexpr int CtlLg = 44;
// 容器高度豁免表（px）
constexpr int HTitleBar  = 36;
constexpr int HNavBar    = 44;   // 导航栏＋导航/模式按钮（原 42 归并）
constexpr int HToolBar   = 56;
constexpr int HFloatBar  = 48;   // 悬浮条内外两层（原 47 归并）
constexpr int HBanner    = 56;   // 近距；远距 64 见 HBannerFar
constexpr int HBannerFar = 64;
constexpr int HValTag    = 20;   // 参数值标签高（去留评审中——先入表保一处改）
// 弹窗宽两族（px）
constexpr int DlgNear = 360; constexpr int DlgFar = 480;
// 近/远档载体（dynamic property；字符串值最稳）
inline constexpr const char* kPropUiFar = "uiFar";
inline constexpr const char* kValNear = "near"; inline constexpr const char* kValFar = "far";
inline int bannerHeight(bool isFar) { return isFar ? HBannerFar : HBanner; }
inline int dlgWidth(bool isFar)    { return isFar ? DlgFar : DlgNear; }
} // namespace UiT
