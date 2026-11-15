$tmp = 'C:\Users\DUUMM\AppData\Local\Temp\opencode\k1b'
function Get-Cycles([string]$f) {
  $d = @{}
  foreach ($l in [System.IO.File]::ReadAllLines($f)) {
    if ($l -match '^\[cycle (\d+)\] (.+)$') { $d[[int]$Matches[1]] = $Matches[2] }
  }
  return $d
}
$a = Get-Cycles "$tmp\abba_A1_host.out"
$b = Get-Cycles "$tmp\abba_B1_m1.out"
$diff = 0
foreach ($k in $a.Keys) { if (-not $b.ContainsKey($k) -or $a[$k] -ne $b[$k]) { $diff++ } }
Write-Output "ABBA oracle host vs m1: $($a.Count) frames, $diff diffs"
# stampsK / fb medians from B1
$rows = @()
foreach ($l in [System.IO.File]::ReadAllLines("$tmp\abba_B1_m1.err")) {
  if ($l -match '^\[v3-time\](.*)$') {
    $h = @{}
    foreach ($m in [regex]::Matches($Matches[1], '([\w-]+)=([-\d.]+)')) { $h[$m.Groups[1].Value] = [double]$m.Groups[2].Value }
    $rows += ,$h
  }
}
foreach ($key in @('stampsK','pairsK','g2','total','fb')) {
  $vals = @(); foreach ($h in $rows) { if ($h.ContainsKey($key)) { $vals += $h[$key] } }
  $s = $vals | Sort-Object
  Write-Output ("  B1 {0,-9} med={1,6:F2}" -f $key, $s[[int](($s.Count - 1) / 2)])
}
