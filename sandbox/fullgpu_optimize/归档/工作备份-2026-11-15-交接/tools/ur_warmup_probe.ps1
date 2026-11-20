# ur_warmup_probe.ps1 — uR 逐线偏差「冷机升温曲线」自动采样与分析
# ============================================================================
# 用途：开机后定时重复测量 25 个逐线 uR 偏差常数（3 分钟复测法的自动化），
#       产出 偏差(t) 曲线——判别「温度→激光模组逐线扇角漂移」假设（定谳档 §8）。
# 前提：speed_replay.exe 已构建；数据集 sparse_30（14 帧，~25 秒/次）。
# 用法（在 fullgpu_optimize 目录下）：
#   冷机开机后立刻启动（默认时刻表见 $Minutes）：
#     powershell -ExecutionPolicy Bypass -File tools\ur_warmup_probe.ps1
#   自定义时刻表（分钟）：
#     ... -Minutes "0,2,4,6,8,10,15,20,30,45,60"
#   只重跑分析（不采样）：
#     ... -Analyze
#   单点冒烟（测一次）：
#     ... -Minutes "0"
# 采样计划（默认）：0,3,6,9,12,15,18,21,24,27,30（快动态段，3 分钟一格）
#                  40,50,60,75,90（趋平衡段，拉稀）——共 16 点，净占用 ~10 分钟。
# 操作规程（重要）：
#   1) 设备冷机 ≥2 小时（最好隔夜断电）后开机，登录后**立即**启动本脚本；
#   2) 采样期间不要跑其他 GPU 任务、不要动设备/标定/表；
#   3) 顺手记录：开机时刻、室温（如有）。GPU 温度脚本自动记（nvidia-smi）；
#   4) 跑完自动出 analysis.txt；若中途断了，-Analyze 可基于已采数据分析。
# 判读标准（定谳档 §8.3）：
#   ① 幅度随时间单调变化并趋于平台 → 温度因果成立；
#   ② 逐线向量与终态向量相关 >0.99（形状不变仅幅度变）→ 单模机制，
#      25 常数可压缩为「1 温度系数 × 固定形状」→ 治本模型可行；
#   ③ 趋平衡时间常数 τ → 产线 SOP 的预热时长依据。
# ============================================================================
param(
    [string]$Minutes = "0,3,6,9,12,15,18,21,24,27,30,40,50,60,75,90",
    [switch]$Analyze,
    [string]$DataDir = "date_in\laser-mark1688-1006\sparse_30",
    [string]$RunDir = ""
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot     # fullgpu_optimize
Set-Location $root
$exe = Join-Path $root 'build-rel\Release\speed_replay.exe'
if (-not (Test-Path $exe)) { throw "speed_replay.exe 不存在：$exe（先构建）" }
if (-not (Test-Path "$root\$DataDir")) { throw "数据目录不存在：$DataDir" }

# —— 运行目录 ——
if (-not $RunDir) {
    $RunDir = Join-Path $root ("out_ur_warmup\run_" + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

# —— 探针配置：清烘表 sidecar（测原始偏差）＋开残差通道（一次性生成）——
$cfgPath = Join-Path $RunDir 'config_probe.json'
if (-not (Test-Path $cfgPath)) {
    $raw = Get-Content (Join-Path $root 'configs\config_baseline.json') -Raw -Encoding UTF8
    $raw = $raw -replace '"cmttBiasSidecar": "\{JMW_SOURCE_ROOT\}/modules/09_operatorlib/TEST/fullgpu_optimize/date_in/laser-mark1688-1006/curve_map_temp_table\.ur_bias\.json"', '"cmttBiasSidecar": ""'
    $raw = $raw -replace '"res_channel": false', '"res_channel": true'
    [System.IO.File]::WriteAllText($cfgPath, $raw, (New-Object System.Text.UTF8Encoding($false)))
    if ($raw -notmatch '"res_channel": true') { throw '配置补丁失败：res_channel 未置 true' }
    Write-Host "[cfg] 探针配置已生成（sidecar 清空 + res_channel=true）：$cfgPath"
}

function Get-GpuTemp {
    try {
        $nsm = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($nsm) { return ((& nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) | Select-Object -First 1) }
    } catch {}
    return ''
}

# —— 单次采样：跑 speed_replay → 解析最新 match_* 的逐线中位 ——
function Invoke-OneSample([int]$Idx, [double]$TargetMin, [datetime]$T0) {
    $pre = @(Get-ChildItem (Join-Path $root 'vis_out') -Directory -Filter 'match_*' -ErrorAction SilentlyContinue | ForEach-Object FullName)
    $env:PATH = 'C:\opencv-cuda-4.13.0\x64\vc17\bin;' + $env:PATH
    # exe 的 stderr（标记点链降级警告等）非致命——EAP 临时放开，跑完为止
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $null = & $exe --config $cfgPath --data-dir $DataDir --match-img --lean --quiet 2>&1 | Out-Null
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $prevEap }
    $newDir = Get-ChildItem (Join-Path $root 'vis_out') -Directory -Filter 'match_*' -ErrorAction SilentlyContinue |
        Where-Object { $pre -notcontains $_.FullName } | Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $newDir) { throw "样本 $Idx 未找到产物目录（exit=$code）" }
    $resFiles = Get-ChildItem $newDir.FullName -Filter '*_res.csv'
    if (-not $resFiles) { throw "样本 $Idx 无 res.csv" }
    $byLid = @{}
    $nTotal = 0
    foreach ($f in $resFiles) {
        foreach ($line in [System.IO.File]::ReadLines($f.FullName) | Select-Object -Skip 1) {
            $c = $line.Split(',')
            $lid = [int]$c[0]; $res = [double]$c[4]
            if (-not $byLid.ContainsKey($lid)) { $byLid[$lid] = New-Object System.Collections.Generic.List[double] }
            $byLid[$lid].Add($res); $nTotal++
        }
    }
    # 保留 res.csv 到样本目录，删 PNG/其余，控磁盘
    $keep = Join-Path $RunDir ("sample_{0:d2}" -f $Idx)
    New-Item -ItemType Directory -Force -Path $keep | Out-Null
    $resFiles | ForEach-Object { Copy-Item $_.FullName $keep -Force }
    Remove-Item $newDir.FullName -Recurse -Force -ErrorAction SilentlyContinue
    # 逐线中位＋斜率
    $lids = $byLid.Keys | Sort-Object
    $med = @{}; $ks = @(); $bs = @()
    foreach ($k in $lids) {
        $a = $byLid[$k].ToArray(); [array]::Sort($a)
        $med[$k] = [math]::Round($a[[int]($a.Length / 2)], 4)
        $ks += [double]$k; $bs += $med[$k]
    }
    $slope = LinearSlope $ks $bs
    $km = @(); $bm = @()
    for ($i = 0; $i -lt $ks.Count; $i++) { if ($ks[$i] -ge 6 -and $ks[$i] -le 22) { $km += $ks[$i]; $bm += $bs[$i] } }
    $slopeMid = LinearSlope $km $bm
    $foot = @(); $tail = @()
    for ($i = 0; $i -lt $ks.Count; $i++) {
        if ($ks[$i] -le 5) { $foot += $bs[$i] } elseif ($ks[$i] -ge 21) { $tail += $bs[$i] }
    }
    $amp = if ($foot.Count -and $tail.Count) {
        ($tail | Measure-Object -Average).Average - ($foot | Measure-Object -Average).Average } else { 0 }
    $now = Get-Date
    $row = [ordered]@{
        idx = $Idx; target_min = $TargetMin
        actual_min = [math]::Round(($now - $T0).TotalMinutes, 2)
        wallclock = $now.ToString('HH:mm:ss'); gpu_temp = Get-GpuTemp
        n_points = $nTotal; slope_all = [math]::Round($slope, 4)
        slope_mid = [math]::Round($slopeMid, 4); amp = [math]::Round($amp, 3)
    }
    foreach ($k in (1..25)) { $row["L$k"] = if ($med.ContainsKey($k)) { $med[$k] } else { '' } }
    return [pscustomobject]$row
}

function LinearSlope($x, $y) {
    $n = $x.Count; if ($n -lt 2) { return 0 }
    $mx = ($x | Measure-Object -Average).Average; $my = ($y | Measure-Object -Average).Average
    $sxy = 0.0; $sxx = 0.0
    for ($i = 0; $i -lt $n; $i++) { $sxy += ($x[$i] - $mx) * ($y[$i] - $my); $sxx += ($x[$i] - $mx) * ($x[$i] - $mx) }
    if ($sxx -le 0) { return 0 }
    return $sxy / $sxx
}

# —— 分析：形状稳定性＋幅度-时间拟合 ——
function Invoke-Analyze {
    $summary = Join-Path $RunDir 'summary.csv'
    if (-not (Test-Path $summary)) { Write-Host "无 summary.csv：$summary"; return }
    $rows = Import-Csv $summary | Sort-Object { [double]$_.actual_min }
    if ($rows.Count -lt 2) { Write-Host "样本不足 2 个，无法分析"; return }
    $lastVec = $rows[-1]
    $rep = @()
    $rep += "=== uR 升温曲线分析（$RunDir）==="
    $rep += ("样本数 {0}；时间跨度 {1}~{2} 分钟" -f $rows.Count, $rows[0].actual_min, $rows[-1].actual_min)
    $rep += ""
    $rep += "idx  t_min  gpuT  slope_mid   amp     corr_with_last"
    foreach ($r in $rows) {
        $corr = ''
        if ($r -ne $lastVec) {
            $xs = @(); $ys = @()
            foreach ($k in (1..25)) {
                $a = [double]$lastVec."L$k"; $b = [double]$r."L$k"
                if ($a -ne '' -and $b -ne '') { $xs += $a; $ys += $b }
            }
            $mx = ($xs | Measure-Object -Average).Average; $my = ($ys | Measure-Object -Average).Average
            $sxy = 0.0; $sxx = 0.0; $syy = 0.0
            for ($i = 0; $i -lt $xs.Count; $i++) {
                $sxy += ($xs[$i] - $mx) * ($ys[$i] - $my); $sxx += ($xs[$i] - $mx) * ($xs[$i] - $mx); $syy += ($ys[$i] - $my) * ($ys[$i] - $my)
            }
            if ($sxx -gt 0 -and $syy -gt 0) { $corr = [math]::Round($sxy / [math]::Sqrt($sxx * $syy), 4) }
        }
        else { $corr = '1.0   (ref)' }
        $rep += ("{0,3}  {1,5}  {2,4}  {3,9}  {4,7}  {5}" -f $r.idx, $r.actual_min, $r.gpu_temp, $r.slope_mid, $r.amp, $corr)
    }
    # 幅度指数趋近拟合：amp(t)=Ainf*(1-exp(-t/tau)) 网格搜 tau
    $ts = @(); $as = @()
    foreach ($r in $rows) { $ts += [double]$r.actual_min; $as += [double]$r.amp }
    $bestTau = 0; $bestA = 0; $bestSse = [double]::MaxValue
    foreach ($tau in (1..120 | ForEach-Object { $_ * 1.0 })) {
        # 线性化：amp = A - A*exp(-t/tau) → 对 A 线性最小二乘
        $se = 0.0; $see = 0.0
        for ($i = 0; $i -lt $ts.Count; $i++) { $e = [math]::Exp(-$ts[$i] / $tau); $se += (1 - $e) * (1 - $e); $see += (1 - $e) * $as[$i] }
        if ($se -le 0) { continue }
        $A = $see / $se; $sse = 0.0
        for ($i = 0; $i -lt $ts.Count; $i++) { $d = $as[$i] - $A * (1 - [math]::Exp(-$ts[$i] / $tau)); $sse += $d * $d }
        if ($sse -lt $bestSse) { $bestSse = $sse; $bestTau = $tau; $bestA = $A }
    }
    $rep += ""
    $rep += ("幅度趋近拟合：amp(t) ≈ {0:N3} × (1 − exp(−t/{1:N0}min))；SSE={2:N4}" -f $bestA, $bestTau, $bestSse)
    $rep += "判读：①幅度单调→温度因果；②corr>0.99→单模（形状固定，可压缩为 1 温度系数）；"
    $rep += "　　　③tau→预热时长参考（SOP：开机后等待或先测一次再烘表）。"
    $txt = $rep -join "`r`n"
    [System.IO.File]::WriteAllText((Join-Path $RunDir 'analysis.txt'), $txt, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host $txt
    Write-Host ""
    Write-Host "已写入 $(Join-Path $RunDir 'analysis.txt')"
}

if ($Analyze) { Invoke-Analyze; exit 0 }

# —— 主采样循环 ——
$sched = $Minutes.Split(',') | ForEach-Object { [double]$_.Trim() } | Sort-Object
$summaryPath = Join-Path $RunDir 'summary.csv'
$header = ((@( 'idx','target_min','actual_min','wallclock','gpu_temp','n_points','slope_all','slope_mid','amp') + (1..25 | ForEach-Object { "L$_" })) -join ',')
if (-not (Test-Path $summaryPath)) { [System.IO.File]::WriteAllText($summaryPath, $header + "`r`n", (New-Object System.Text.UTF8Encoding($false))) }
$T0 = Get-Date
Write-Host "[plan] 启动于 $($T0.ToString('HH:mm:ss'))；时刻表(分)：$($sched -join ', ')；产物：$RunDir"
$idx = 0
foreach ($m in $sched) {
    $target = $T0.AddMinutes($m)
    $wait = ($target - (Get-Date)).TotalSeconds
    if ($wait -gt 0) { Write-Host ("[wait] 下一样本 t+{0}min（{1}s 后）" -f $m, [int][math]::Ceiling($wait)); Start-Sleep -Seconds ([int][math]::Ceiling($wait)) }
    try {
        $row = Invoke-OneSample -Idx $idx -TargetMin $m -T0 $T0
        $line = ($row.PSObject.Properties | ForEach-Object { $_.Value }) -join ','
        [System.IO.File]::AppendAllText($summaryPath, $line + "`r`n")
        Write-Host ("[sample {0}] t+{1}min gpuT={2} slope_mid={3} amp={4} n={5}" -f $idx, $row.actual_min, $row.gpu_temp, $row.slope_mid, $row.amp, $row.n_points)
    }
    catch { Write-Warning "样本 $idx 失败：$($_.Exception.Message)（继续）" }
    $idx++
}
Invoke-Analyze
Write-Host "[done] 全部完成。冷机实验判读后把 analysis.txt 与结论归档（定谳档 §8.3 实验①）。"
