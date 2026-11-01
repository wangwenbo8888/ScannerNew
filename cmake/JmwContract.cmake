# ============================================================================
# JmwContract.cmake — jmw_add_contract(<target> <prefix> <module_src_dir>
#                                      EXTRA_DIRS <dir>...)
#
# 261031 D1-lite：模块契约指纹机制（用户口径：编译 07/09 两个 DLL 直接
# 拷贝替换即可更新算法）。为模块 DLL 建立「公共头闭包指纹」：
#
#   指纹源 = 工具链串(msvc/crt/opencv) ＋ 全部闭包头内容
#     · 本模块源码树 *.h（排除 tests/、TEST/、*_pimpl.h）
#     · EXTRA_DIRS 指定的传递依赖公共头（只收根层级 *.h，不递归）
#       —— 09 闭包＝自身＋base；07 闭包＝自身＋base＋06＋09
#   产物①  <binary>/jmw_<prefix>_contract_hash.h —— constexpr 指纹/ABI，
#           经 PUBLIC include 同时供「本模块导出实现」与「消费方编译期
#           期望值」引用（同一份，两侧天然对齐）
#   产物②  api/<prefix>_contract.cpp —— extern "C" 导出
#           jmw_<prefix>_abi_version() / jmw_<prefix>_contract_hash()
#
# 运行时守护链：exe 启动校验 07；07 会话配置校验 09——
#   只改实现 → 指纹不变 → 换 DLL 直接生效（算法热更）
#   改闭包内任一头 / 换工具链 → 指纹变 → 拒绝启动并明示全量重编
# ============================================================================
function(jmw_add_contract TGT PREFIX MOD_DIR)
  set(_extra "")
  set(_in_extra FALSE)
  foreach(_a IN LISTS ARGN)
    if(_a STREQUAL "EXTRA_DIRS")
      set(_in_extra TRUE)
    elseif(_in_extra)
      list(APPEND _extra "${_a}")
    endif()
  endforeach()

  file(GLOB_RECURSE _hdrs "${MOD_DIR}/*.h")
  set(_filtered "")
  foreach(_h IN LISTS _hdrs)
    string(TOLOWER "${_h}" _low)
    if(_low MATCHES "(^|[\\/])tests?[\\/]" OR _h MATCHES "_pimpl")
      continue()
    endif()
    list(APPEND _filtered "${_h}")
  endforeach()
  # EXTRA_DIRS 递归收头，同一套排除规则——06/09 整树头本就是其 ABI 面，
  # 全量冻结语义正确；base 亦同
  foreach(_d IN LISTS _extra)
    file(GLOB_RECURSE _eh "${_d}/*.h")
    foreach(_h IN LISTS _eh)
      string(TOLOWER "${_h}" _low)
      if(_low MATCHES "(^|[\\/])tests?[\\/]" OR _h MATCHES "_pimpl")
        continue()
      endif()
      list(APPEND _filtered "${_h}")
    endforeach()
  endforeach()
  list(SORT _filtered)
  list(REMOVE_DUPLICATES _filtered)

  set(_list "${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_contract_headers.txt")
  string(REPLACE ";" "\n" _list_txt "${_filtered}")
  file(WRITE "${_list}" "${_list_txt}")

  set(_toolchain "msvc=${MSVC_VERSION};crt=${CMAKE_MSVC_RUNTIME_LIBRARY};opencv=${OpenCV_VERSION}")
  set(_gen "${CMAKE_CURRENT_BINARY_DIR}/jmw_${PREFIX}_contract_hash.h")
  add_custom_command(OUTPUT "${_gen}"
    COMMAND "${CMAKE_COMMAND}"
            -DJMW_PREFIX=${PREFIX}
            "-DJMW_TOOLCHAIN=${_toolchain}"
            -DJMW_HDR_LIST=${_list}
            -DJMW_OUT=${_gen}
            -P "${CMAKE_SOURCE_DIR}/cmake/contract_hash.cmake"
    DEPENDS ${_filtered} ${_list}
    COMMENT "[${TGT}] 契约指纹 ${PREFIX}")
  target_sources(${TGT} PRIVATE
    "api/${PREFIX}_contract.cpp"
    "${_gen}")
  target_include_directories(${TGT} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
endfunction()
