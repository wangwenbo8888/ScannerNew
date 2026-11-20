#!/bin/bash
# V5 参数扫参（顺序单参数切）——快扫口径：满25线帧数([v5:mask])＋总配对(cycle行laser=)
# 用法: bash sweep.sh   （在 fullgpu_optimize 根目录下）
set -u
BASE=configs/variants/v5_sweep_base.json
OUT=/tmp/v5sw
mkdir -p "$OUT"
EXE=./build-rel/Release/speed_replay.exe
DATA=date_in/laser-mark1688-1006/ALL_raw

run_round() {  # $1=tag  其余=sed 表达式(作用于 v5 节唯一键)
  local tag="$1"; shift
  local cfg="$OUT/cfg_$tag.json"
  cp "$BASE" "$cfg"
  local e
  for e in "$@"; do sed -i "$e" "$cfg"; done
  local d="$OUT/r_$tag"; rm -rf "$d"; mkdir -p "$d"
  local t0=$(date +%s)
  JMW_LMV5=1 JMW_V5DIAG=1 "$EXE" --config "$cfg" --data-dir "$DATA" \
    --cycles 0 --repeat 1 --lean --quiet --dump-dir "$d" > "$d/out.log" 2> "$d/err.log"
  local t1=$(date +%s)
  local cov=$(grep -c '\[v5:mask\] 25 ' "$d/err.log")
  local all=$(grep -c '\[v5:mask\]' "$d/err.log")
  local pairs=$(grep -oE 'laser=[0-9]+' "$d/out.log" | cut -d= -f2 | awk '{s+=$1} END {print s}')
  local fps=""
  [ -f "$d/fps_summary.csv" ] && fps=$(grep '^ALL,' "$d/fps_summary.csv" | cut -d, -f3 | head -1)
  echo "$tag  cov=$cov/$all  pairs=$pairs  fps=$fps  secs=$((t1-t0))"
}

echo "== baseline =="
run_round base
echo "== share_fast =="
run_round sh70 -e 's/"share_fast": 0.8/"share_fast": 0.7/'
run_round sh60 -e 's/"share_fast": 0.8/"share_fast": 0.6/'
run_round sh90 -e 's/"share_fast": 0.8/"share_fast": 0.9/'
echo "== margin_min =="
run_round mg15 -e 's/"margin_min": 2.0/"margin_min": 1.5/'
run_round mg12 -e 's/"margin_min": 2.0/"margin_min": 1.2/'
run_round mg30 -e 's/"margin_min": 2.0/"margin_min": 3.0/'
echo "== votes_floor =="
run_round vf2 -e 's/"votes_floor": 4/"votes_floor": 2/'
run_round vf8 -e 's/"votes_floor": 4/"votes_floor": 8/'
echo "== quality_e_min / log / depth / clean =="
run_round e15 -e 's/"quality_e_min": 1.9/"quality_e_min": 1.5/'
run_round log1 -e 's/"vote_weight_log": false/"vote_weight_log": true/'
run_round dep0 -e 's/"depth_gate": true/"depth_gate": false/'
run_round cln0 -e 's/"res_clean": true/"res_clean": false/'
