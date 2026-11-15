#!/usr/bin/env python3
"""每隔30帧取一帧，输出每段的票数+结果分析表"""
import re, sys

with open("/tmp/v5z/votes.log", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

# 解析 votes + vres 按帧分组
frames = []  # [(votes_list, vres_list)]
cur_votes, cur_vres = [], []
last_was_votes = True
for line in lines:
    if "[v5:votes]" in line:
        if last_was_votes is False:  # 新帧开始
            frames.append((cur_votes, cur_vres))
            cur_votes, cur_vres = [], []
        cur_votes.append(line.strip())
        last_was_votes = True
    elif "[v5:vres]" in line:
        cur_vres.append(line.strip())
        last_was_votes = False
if cur_votes:
    frames.append((cur_votes, cur_vres))

print(f"总帧数: {len(frames)}", file=sys.stderr)

# 目标帧
targets = list(range(0, len(frames), 30))
if targets[-1] != len(frames)-1:
    targets.append(len(frames)-1)

vote_re = re.compile(r'seg=(\d+) trk=(-?\d+) pts=(\d+) carded=(\d+) x~([\d.-]+) y([\d.-]+)-([\d.-]+) drop=(\d+) \|(.*)')
lid_re  = re.compile(r'L(\d+):(\d+)\((\d+)%(\*?)\)')
res_re  = re.compile(r'seg=(\d+) winner=(-?\d+) drop=(\d+)')

for fi in targets:
    votes, vres = frames[fi]
    # winner 映射
    wmap = {}
    for vl in vres:
        m = res_re.search(vl)
        if m:
            wmap[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))

    # 解析每个有票段
    segs = []
    for vl in votes:
        m = vote_re.search(vl)
        if not m or int(m.group(3)) == 0:
            continue
        seg = int(m.group(1)); pts = int(m.group(3)); x = float(m.group(5))
        y1 = float(m.group(6)); y2 = float(m.group(7))
        # 解析票
        lids = []
        for lm in lid_re.finditer(m.group(10)):
            lids.append((int(lm.group(1)), int(lm.group(2)), int(lm.group(3)), lm.group(4)=="*"))
        lids.sort(key=lambda t: -t[1])
        w, d = wmap.get(seg, (-1, -1))
        segs.append((seg, pts, x, y1, y2, lids, w, d))

    print(f"\n## 帧 #{fi} ({len(segs)} 个有票段)")
    print(f"| 段 | 点数 | 位置x | 第一名(票数/占比) | 第二名(票数/占比) | 第三名(票数/占比) | 其余线号数 | 判定 | 弃因 |")
    print(f"|---|---|---|---|---|---|---|---|---|")

    for seg, pts, x, y1, y2, lids, w, d in sorted(segs, key=lambda s: -s[1]):
        cands = [l for l in lids if l[1] > 0]
        n_other = len(cands) - 3 if len(cands) > 3 else 0
        s1 = f"L{cands[0][0]}:{cands[0][1]}({cands[0][2]}%{'*' if cands[0][3] else ''})" if len(cands)>0 else "—"
        s2 = f"L{cands[1][0]}:{cands[1][1]}({cands[1][2]}%{'*' if cands[1][3] else ''})" if len(cands)>1 else "—"
        s3 = f"L{cands[2][0]}:{cands[2][2]}%" if len(cands)>2 else "—"
        others = f"+{n_other}线" if n_other > 0 else ""
        drop_str = {0:"", 1:"无卡", 2:"无支持", 3:"消元", 4:"兜底"}.get(d, str(d))
        print(f"| seg{seg} | {pts} | {x:.0f} | {s1} | {s2} | {s3}{others} | {w} | {drop_str} |")

    # 统计摘要
    big = [s for s in segs if s[1] >= 60]
    small = [s for s in segs if s[1] < 60]
    ok = [s for s in segs if s[6] >= 1]
    dropped = [s for s in segs if s[7] != 0]
    # 第一名占比分布
    top_shares = []
    close_pairs = []  # 第一二名接近(<1.05倍)的段
    for seg, pts, x, y1, y2, lids, w, d in segs:
        if len(lids) >= 2 and lids[0][1] > 0:
            top_shares.append(lids[0][2])
            if lids[1][1] > 0:
                ratio = lids[0][1] / lids[1][1]
                if ratio < 1.05:
                    close_pairs.append((seg, pts, lids[0][0], lids[0][1], lids[1][0], lids[1][1], w))
    if top_shares:
        ts = sorted(top_shares)
        print(f"\n**摘要**: 大段{len(big)}/小段{len(small)} | 定{len(ok)}段/弃{len(dropped)}段 | "
              f"第一名占比 min={ts[0]}% / 中位={ts[len(ts)//2]}% / max={ts[-1]}% | "
              f"一二名接近(<1.05倍): {len(close_pairs)}段")
        if close_pairs:
            print(f"  接近段: ", end="")
            for cp in close_pairs[:5]:
                print(f"seg{cp[0]}({cp[1]}pts L{cp[2]}:{cp[3]} vs L{cp[4]}:{cp[5]} →W{cp[6]}) ", end="")
            if len(close_pairs) > 5:
                print(f"...共{len(close_pairs)}段", end="")
            print()
