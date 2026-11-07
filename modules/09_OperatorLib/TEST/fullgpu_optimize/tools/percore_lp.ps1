# percore_lp.ps1 - per-LP CPU utilization profile during one run (ASCII only)
# Usage: percore_lp.ps1 -Exe <speed_replay> [-Config <json>] [-Repeat 3] [-Tag <name>]
# Output: C:\Users\DUUMM\AppData\Local\Temp\opencode\percore\<Tag>\{lp.csv,fps_summary.csv} + console table
# Pitfalls (2026-10-29 all stepped on): typeperf -o REFUSES to overwrite a stale
# output file ("no valid counters") -> always delete before sampling; load-window
# self-filter = keep samples with 12-LP total >= 150 (startup/rehash dilution guard);
# PowerShell pipeline early-close (Select -First N on native output) kills the run.
param(
  [Parameter(Mandatory=$true)][string]$Exe,
  [string]$Config = "E:\JEAMMWARE2601001\modules\09_operatorlib\TEST\fullgpu_optimize\configs\config_baseline.json",
  [int]$Repeat = 3,
  [string]$Tag = "run"
)
$env:PATH = 'C:\opencv-cuda-4.13.0\x64\vc17\bin;' + $env:PATH
$out = "C:\Users\DUUMM\AppData\Local\Temp\opencode\percore\$Tag"
New-Item -ItemType Directory -Force $out | Out-Null
$cf = "$out\counters.txt"
@((0..11 | ForEach-Object { "\Processor($_)\% Processor Time" }), '\Processor Information(_Total)\% Processor Performance') | Set-Content -Path $cf -Encoding ASCII
$tf = "$out\lp.csv"
Remove-Item $tf -Force -ErrorAction SilentlyContinue
$job = Start-Job -ScriptBlock { param($a,$b,$c) typeperf -cf $a -si 1 -f CSV -o $b -sc $c } -ArgumentList $cf, $tf, 70
& $Exe --config $Config --cycles 0 --repeat $Repeat --lean --quiet --dump-dir $out *> $null
Wait-Job $job -Timeout 20 | Out-Null
Remove-Job $job -Force -ErrorAction SilentlyContinue
$acc = @{}
foreach ($ln in (Get-Content $tf | Where-Object { $_ -match '^"' })) {
  $v = $ln -replace '"','' -split ','
  if ($v.Count -ge 13) {
    $s = 0.0; $ok = $true
    for ($i = 1; $i -le 12; $i++) { if ($v[$i] -match '^\d') { $s += [double]$v[$i] } else { $ok = $false } }
    if ($ok -and $s -ge 150) { for ($i = 1; $i -le 12; $i++) { $acc[$i] = @($acc[$i]) + [double]$v[$i] } }
  }
}
$fps = [double]((Import-Csv "$out\fps_summary.csv" -Encoding UTF8 | Where-Object { $_.pass -eq 'ALL' }).median_fps)
$labels = @('LP0 sys','LP1 ch2','LP2 1a','LP3 sib','LP4 1b','LP5 sib','LP6 Mk1','LP7 sib','LP8 Mk2','LP9 sib','LP10 Mk3','LP11 sib')
Write-Output ("fps={0}  frame={1}ms" -f $fps, [math]::Round(1000/$fps,2))
for ($i = 1; $i -le 12; $i++) {
  $c = @($acc[$i] | Sort-Object); if ($c.Count -eq 0) { continue }
  $m = $c[[int](($c.Count-1)/2)]
  Write-Output ("{0}  {1,5:F1}%  {2,5:F2} ms/frame" -f $labels[$i-1], $m, ($m/100*1000/$fps))
}
