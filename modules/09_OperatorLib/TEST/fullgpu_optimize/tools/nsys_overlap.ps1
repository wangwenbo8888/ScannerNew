# nsys_overlap.ps1 - kernel collision analysis from gpu_trace CSV (ASCII only)
# Usage: nsys_overlap.ps1 -TraceCsv <m0_trace.csv> -Kernel fuseKernel [-Skip 2000]
#        nsys_overlap.ps1 -TraceCsv <csv> -Kernel fuseKernel -Partner
# -Partner: also lists top collision partners by overlapped time.
# Trace CSV from: nsys stats --report cuda_gpu_trace --format csv -o <prefix> <rep>
param(
  [Parameter(Mandatory=$true)][string]$TraceCsv,
  [Parameter(Mandatory=$true)][string]$Kernel,
  [int]$Skip = 2000,
  [switch]$Partner
)
$rows = Import-Csv $TraceCsv -Encoding UTF8
$all = @($rows | Where-Object { $_.'Start (ns)' -match '^\d' } | Select-Object -Skip $Skip)
$tgt = @($all | Where-Object { $_.'Name' -match $Kernel } | Sort-Object { [double]$_.'Start (ns)' })
$others = @($all | Where-Object { $_.'Name' -notmatch $Kernel } | Sort-Object { [double]$_.'Start (ns)' })
$ovRatios = @(); $durs = @(); $partnerTime = @{}
$oIdx = 0
foreach ($tk in $tgt) {
  $ts = [double]$tk.'Start (ns)'; $td = [double]$tk.'Duration (ns)'; $te = $ts + $td
  $durs += $td; $ov = 0.0
  while ($oIdx -lt $others.Count -and (([double]$others[$oIdx].'Start (ns)' + [double]$others[$oIdx].'Duration (ns)') -lt $ts)) { $oIdx++ }
  for ($j = $oIdx; $j -lt $others.Count; $j++) {
    $os = [double]$others[$j].'Start (ns)'; $oe = $os + [double]$others[$j].'Duration (ns)'
    if ($os -gt $te) { break }
    $s = [Math]::Max($ts, $os); $e = [Math]::Min($te, $oe)
    if ($e -gt $s) {
      $ov += ($e - $s)
      if ($Partner) { $nm = ($others[$j].'Name' -split '\(')[0]; $partnerTime[$nm] = [double]$partnerTime[$nm] + ($e - $s) }
    }
  }
  $ovRatios += ($ov / $td)
}
$n = $tgt.Count
$r = @($ovRatios | Sort-Object); $d = @($durs | Sort-Object)
Write-Output ("{0}: n={1}  durMed={2}ns avg={3}ns  overlapAny_med={4}%" -f $Kernel, $n, $d[[int](($n-1)/2)], [math]::Round(($durs | Measure-Object -Average).Average,0), [math]::Round($r[[int](($n-1)/2)]*100,1))
if ($Partner) {
  $tot = 0.0; foreach ($k in $partnerTime.Keys) { $tot += $partnerTime[$k] }
  foreach ($e2 in ($partnerTime.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 8)) {
    Write-Output ("{0,10:F1} ms  {1,5:F1}%  {2}" -f ($e2.Value/1000000.0), ($e2.Value/$tot*100), $e2.Key)
  }
}
