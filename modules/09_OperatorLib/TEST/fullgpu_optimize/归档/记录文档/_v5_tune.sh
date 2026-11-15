#!/bin/bash
# V5 全自动参数搜索（坐标上升）：目标 399/399 满 25 线；只改配置不改代码。
# 口径：串行+JMW_V5DIAG 快扫（覆盖=[v5:mask] 25 帧数；配对=cycle行 laser= 求和）。
# 状态覆盖集以 "k=v ..." 字符串承载；每轮从 config_baseline 重新生成串行基，再套覆盖。
set -u
cd "E:/JEAMMWARE260705/modules/09_OperatorLib/TEST/fullgpu_optimize"
EXE=./build-rel/Release/speed_replay.exe
DATA=date_in/laser-mark1688-1006/ALL_raw
W=/tmp/v5tune; rm -rf $W; mkdir -p $W
TSV=$W/results.tsv; : > $TSV
NEVAL=0

mkbase() { sed 's/"executor": "fivechain"/"executor": "serial"/' configs/config_baseline.json > $W/base.json; }

set_param() { # file key val（仅作用于 v5 节——行前缀纯空白且键精确匹配，排除 _c_ 注释行）
  awk -v k="$2" -v v="$3" '
    /"laser_match_scan_v5": \{/ {inv5=1}
    /"laser_reconstruct": \{/ {inv5=0}
    {
      if (inv5) {
        i = index($0, "\"" k "\":")
        if (i >= 1) {
          pre = substr($0, 1, i - 1)
          if (pre ~ /^ *$/) sub(/:[^,]*/, ": " v, $0)
        }
      }
      print
    }' "$1" > "$1.tmp" && mv "$1.tmp" "$1"
}

getcur() { # state-str key -> val（缺省回 config 值由 DEFAULTS 提供）
  local st="$1" k="$2"
  for tok in $st; do [ "${tok%%=*}" = "$k" ] && { echo "${tok#*=}"; return; }; done
  eval "echo \${DEF_$k}"
}

evalov() { # tag state -> 输出 "cov pairs"，记录 TSV
  local tag="$1" st="$2"
  mkbase
  local kv k v
  for kv in $st; do k="${kv%%=*}"; v="${kv#*=}"; set_param $W/base.json "$k" "$v"; done
  local d=$W/run; rm -rf $d; mkdir -p $d
  JMW_LMV5=1 JMW_V5DIAG=1 $EXE --config $W/base.json --data-dir $DATA \
    --cycles 0 --repeat 1 --lean --quiet --dump-dir $d > $d/out.log 2> $d/err.log
  local cov=$(grep -c '\[v5:mask\] 25 ' $d/err.log)
  local pairs=$(grep -oE 'laser=[0-9]+' $d/out.log | cut -d= -f2 | awk '{s+=$1}END{print s+0}')
  NEVAL=$((NEVAL+1))
  echo -e "$tag\t$cov\t$pairs\t$st" >> $TSV
  echo "[$NEVAL] $tag cov=$cov pairs=$pairs"
}

better() { # cov1 pairs1 cov2 pairs2 —— 覆盖优先，配对破平
  [ "$1" -gt "$3" ] && return 0
  [ "$1" -eq "$3" ] && [ "$2" -gt "$4" ] && return 0
  return 1
}

# ── 当前定稿（config 值即搜索起点） ──
DEF_share_fast=0.8; DEF_margin_min=1.2; DEF_votes_floor=4
DEF_quality_e_min=1.9; DEF_quality_sep_px=0.5; DEF_match_threshold=4.0
DEF_vL_tolerance=0.6; DEF_depth_gate=true

BEST=""; evalov base "$BEST"; read -r _ BC BP _ <<< "$(tail -1 $TSV | tr '\t' ' ')"

DIMS="share_fast margin_min votes_floor quality_e_min quality_sep_px match_threshold vL_tolerance depth_gate"
VALS_share_fast="0.6 0.7 0.8"
VALS_margin_min="1.0 1.1 1.2"
VALS_votes_floor="1 2 4"
VALS_quality_e_min="1.5 1.9"
VALS_quality_sep_px="0.35 0.5"
VALS_match_threshold="4.0 5.0 6.0 7.0"
VALS_vL_tolerance="0.6 0.8"
VALS_depth_gate="true false"

for sweep in 1 2 3; do
  improved=0
  for dim in $DIMS; do
    cur=$(getcur "$BEST" $dim)
    for v in $(eval echo \${VALS_$dim}); do
      [ "$v" = "$cur" ] && continue
      st="$BEST $dim=$v"
      evalov "s$sweep-$dim=$v" "$st"
      read -r _ C P _ <<< "$(tail -1 $TSV | tr '\t' ' ')"
      if better $C $P $BC $BP; then BEST="$st"; BC=$C; BP=$P; improved=1
        echo "  ↑ 新最优 cov=$BC pairs=$BP"; fi
      [ $BC -ge 399 ] && break 3
    done
  done
  echo "== sweep$sweep 完成：cov=$BC pairs=$BP best=[$BEST] =="
  [ $improved -eq 0 ] && break
done

echo "===== 搜索结束 ====="
echo "BEST_STATE=$BEST"
echo "BEST_COV=$BC BEST_PAIRS=$BP  (评估轮数=$NEVAL)"
sort -t$'\t' -k2,2nr -k3,3nr $TSV | head -8

# ── 赢家复验两遍（防单轮抖动） ──
evalov verify1 "$BEST"; evalov verify2 "$BEST"
echo "VERIFY_DONE"
