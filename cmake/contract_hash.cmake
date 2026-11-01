# ============================================================================
# contract_hash.cmake — 构建期契约指纹生成（cmake -P 脚本）
#
# 用法：
#   cmake -DJMW_PREFIX=op -DJMW_TOOLCHAIN=<工具链串>
#         -DJMW_HDR_LIST=<头列表文件> -DJMW_OUT=<输出头> -P 本脚本
#
# 指纹 = 工具链串 + 闭包内全部公共头内容（按路径排序拼接）的 SHA-256
# （261031 D1-lite：07/09 DLL 单独替换的 ABI 守护）。
#   · 闭包 = 本模块公共头 ＋ 传递依赖头（base/06/09，由调用方给全）
#     ——ScanPipeline.h 直接 include 06 的 SlotRing/EnhancedFrame 与 09 的
#     calib_result_types.h，故 07 的导出 ABI 由多层头共同构成，只哈希自身
#     会漏检
#   · 工具链串 = MSVC 版本＋CRT＋OpenCV 版本——STL/cv::Mat 跨界（C2 现状）
#     依赖同工具链，换库版本必须被看见
# 输出内容不变则不落盘（避免无谓的重建级联）。
# ============================================================================
file(STRINGS "${JMW_HDR_LIST}" _headers)
set(_blob "${JMW_TOOLCHAIN}")
foreach(_h IN LISTS _headers)
  file(READ "${_h}" _c)
  string(APPEND _blob "${_c}")
endforeach()
string(SHA256 _hash "${_blob}")
set(_content "#pragma once
// 生成文件（构建期契约指纹）——勿手改；变更请重编所属模块与其消费方
#include <cstdint>
namespace jmw::contract::${JMW_PREFIX} {
inline constexpr const char* hash = \"${_hash}\";
inline constexpr uint32_t abi_version = 1;
}
")
if(EXISTS "${JMW_OUT}")
  file(READ "${JMW_OUT}" _old)
  if(_old STREQUAL _content)
    return()
  endif()
endif()
file(WRITE "${JMW_OUT}" "${_content}")
message(STATUS "[${JMW_PREFIX}] 契约指纹 ${_hash}")
