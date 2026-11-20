# ladder_ab.ps1 - multi-arm same-window interleaved ladder (ASCII only)
# Usage: ladder_ab.ps1 -ExeA <a.exe> -LabelA A [-ExeB <b.exe>] [-LabelB B]
#        [-ExeC <c.exe>] [-LabelC C] [-Blocks 3] [-Repeat 3] [-EnvB "K=V;K2=V2"]
# Same-window discipline: arms interleave within each block; only intra-block
# comparisons are valid. Cross-window magnitude comparison is BANNED (three
# proven traps on 2026-10-29). Optionally prints per-5s fps buckets per run.
param(
  [Parameter(Mandatory=$true)][string]$ExeA,
  [string]$LabelA = "A",
  [string]$ExeB = "",
  [string]$LabelB = "B",
  [string]$ExeC = "",
  [string]$LabelC = "C",
  [int]$Blocks = 3,
  [int]$Repeat = 3,
  [string]$EnvB = "",
  [string]$Config = "E:\JEAMMWARE2601001\modules\09_operatorlib\TEST\fullgpu_optimize\configs\config_baseline.json"
)
$env:PATH = 'C:\opencv-cuda-4.13.0\x64\vc17\bin;' + $env:PATH
$root = "C:\Users\DUUMM\AppData\Local\Temp\opencode\ladder_$(Get-Date -Format HHmmss)"
$arms = [ordered]@{ $LabelA = $ExeA }
if ($ExeB) { $arms[$LabelB] = $ExeB }
if ($ExeC) { $arms[$LabelC] = $ExeC }
$res = @{}
foreach ($blk in 1..$Blocks) {
  foreach ($k in $arms.Keys) {
    $dir = "$root\b${blk}_$k"; New-Item -ItemType Directory -Force $dir | Out-Null
    if ($k -eq $LabelB -and $EnvB) {
      foreach ($kv in ($EnvB -split ';')) { $p = $kv -split '='; Set-Item "Env:$($p[0])" $p[1] }
    }
    & $arms[$k] --config $Config --cycles 0 --repeat $Repeat --lean --quiet --dump-dir $dir *> $null
    if ($k -eq $LabelB -and $EnvB) { foreach ($kv in ($EnvB -split ';')) { $p = $kv -split '='; Remove-Item "Env:$($p[0])" -ErrorAction SilentlyContinue } }
    $med = [double]((Import-Csv "$dir\fps_summary.csv" -Encoding UTF8 | Where-Object { $_.pass -eq 'ALL' }).median_fps)
    $res["b$blk$k"] = $med
    Write-Output ("b{0} {1,-6} median={2}" -f $blk, $k, $med)
  }
}
Write-Output "== per-arm median-of-blocks =="
foreach ($k in $arms.Keys) {
  $m = @(1..$Blocks | ForEach-Object { $res["b$_$k"] } | Sort-Object)
  Write-Output ("{0,-6}: {1}  => median {2}" -f $k, ($m -join ' / '), $m[[int](($Blocks-1)/2)])
}
