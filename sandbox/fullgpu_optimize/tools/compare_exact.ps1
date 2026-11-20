# compare_exact.ps1 — 字节级精确对拍（金样 report.csv 等）：MD5 逐文件比对
# 缺文件/哈希不等 → FAIL。用于"纯重构必须零变化"的硬门禁（compare_run.ps1 是容差门，别混用）。
# 用法: powershell -NoProfile -ExecutionPolicy Bypass -File tools/compare_exact.ps1 -Golden <file> -Candidate <file> [-Label <tag>]
param(
    [Parameter(Mandatory = $true)][string]$Golden,
    [Parameter(Mandatory = $true)][string]$Candidate,
    [string]$Label = 'file'
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Golden))    { Write-Host "[compare_exact] FAIL: golden missing: $Golden"; exit 1 }
if (-not (Test-Path -LiteralPath $Candidate)) { Write-Host "[compare_exact] FAIL: candidate missing: $Candidate"; exit 1 }

$gh = (Get-FileHash -Algorithm MD5 -LiteralPath $Golden).Hash
$ch = (Get-FileHash -Algorithm MD5 -LiteralPath $Candidate).Hash

# 同字节数不同内容时给出首差异行定位（便于归因）
if ($gh -ne $ch) {
    $gl = [System.IO.File]::ReadAllLines($Golden)
    $cl = [System.IO.File]::ReadAllLines($Candidate)
    Write-Host ("[compare_exact] FAIL: {0} md5 {1} vs {2} (lines {3} vs {4})" -f $Label, $gh, $ch, $gl.Count, $cl.Count)
    $n = [Math]::Min($gl.Count, $cl.Count)
    for ($i = 0; $i -lt $n; $i++) {
        if ($gl[$i] -cne $cl[$i]) { Write-Host ("[compare_exact] first diff line {0}:  gold={1}  cand={2}" -f ($i + 1), $gl[$i], $cl[$i]); break }
    }
    exit 1
}
Write-Host "[compare_exact] $Label md5=$gh -> PASS (byte-exact)"
exit 0
