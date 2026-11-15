# compare_res.ps1 — G4 逐点对拍：两目录的 frame_NNN_res.csv 逐帧逐行比对
# 第一判据：任一侧目录缺失 / _res.csv 计数不等 / 任帧缺失 → FAIL（防"零匹配帧缺文件被放过"）
# 用法: powershell -NoProfile -ExecutionPolicy Bypass -File tools/compare_res.ps1 -Baseline <dir> -Candidate <dir>
param(
    [Parameter(Mandatory = $true)][string]$Baseline,
    [Parameter(Mandatory = $true)][string]$Candidate
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Baseline)) { Write-Host "[compare_res] FAIL: baseline dir missing: $Baseline"; exit 1 }
if (-not (Test-Path -LiteralPath $Candidate)) { Write-Host "[compare_res] FAIL: candidate dir missing: $Candidate"; exit 1 }

$bFiles = @(Get-ChildItem -LiteralPath $Baseline -Filter '*_res.csv' | Sort-Object Name)
$cFiles = @(Get-ChildItem -LiteralPath $Candidate -Filter '*_res.csv' | Sort-Object Name)

if ($bFiles.Count -eq 0) { Write-Host "[compare_res] FAIL: baseline has no _res.csv"; exit 1 }
if ($bFiles.Count -ne $cFiles.Count) {
    Write-Host "[compare_res] FAIL: res.csv count mismatch baseline=$($bFiles.Count) candidate=$($cFiles.Count)"; exit 1
}

$bMap = @{}; foreach ($f in $bFiles) { $bMap[$f.Name] = $f.FullName }
$cMap = @{}; foreach ($f in $cFiles) { $cMap[$f.Name] = $f.FullName }

$diffs = 0; $checked = 0
foreach ($name in $bMap.Keys) {
    if (-not $cMap.ContainsKey($name)) { Write-Host "[compare_res] FAIL: candidate missing $name"; exit 1 }
    $checked++
    $bl = [System.IO.File]::ReadAllLines($bMap[$name])
    $cl = [System.IO.File]::ReadAllLines($cMap[$name])
    if ($bl.Count -ne $cl.Count) { Write-Host "[compare_res] FAIL: $name line count $($bl.Count) vs $($cl.Count)"; $diffs++; continue }
    $n = $bl.Count
    for ($i = 0; $i -lt $n; $i++) {
        if ($bl[$i] -cne $cl[$i]) {
            if ($diffs -lt 5) { Write-Host ("[compare_res] DIFF {0}:{1}  base={2}  cand={3}" -f $name, ($i + 1), $bl[$i], $cl[$i]) }
            $diffs++
            if ($diffs -ge 1000) { Write-Host "[compare_res] (>=1000 diffs, truncating)"; break }
        }
    }
}

# 多出的候选文件（基线没有的帧）也算 FAIL
foreach ($name in $cMap.Keys) { if (-not $bMap.ContainsKey($name)) { Write-Host "[compare_res] FAIL: candidate extra $name"; exit 1 } }

if ($diffs -gt 0) { Write-Host "[compare_res] checked=$checked diffs=$diffs -> FAIL"; exit 1 }
Write-Host "[compare_res] checked=$checked diffs=0 (byte-exact per line) -> PASS"
exit 0
