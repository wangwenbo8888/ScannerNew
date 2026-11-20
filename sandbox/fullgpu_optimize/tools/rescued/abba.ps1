$ErrorActionPreference = 'Stop'
$so = 'E:\JEAMMWARE2601001\modules\09_operatorlib\TEST\speed_optimize'
$tmp = 'C:\Users\DUUMM\AppData\Local\Temp\opencode\k1b'
$env:JMW_SOURCE_ROOT = 'E:\JEAMMWARE2601001'
$env:PATH = 'C:\opencv-cuda-4.13.0\x64\vc17\bin;' + $env:PATH
$seq = @(
  @{ t = 'A1_host'; e = @{} },
  @{ t = 'B1_m1';   e = @{ 'JMW_V3FULLGPU' = '1' } },
  @{ t = 'B2_m1';   e = @{ 'JMW_V3FULLGPU' = '1' } },
  @{ t = 'A2_host'; e = @{} }
)
foreach ($r in $seq) {
  foreach ($k in $r.e.Keys) { Set-Item -Path "Env:$k" -Value $r.e[$k] }
  cmd /c "cd /d E:\JEAMMWARE2601001\build-rel\so\Release && speed_replay.exe --config `"$so\config_all.json`" --lean --quiet --dump-dir `"$tmp\out_abba_$($r.t)`" 1>$tmp\abba_$($r.t).out 2>$tmp\abba_$($r.t).err" | Out-Null
  foreach ($k in $r.e.Keys) { Remove-Item -Path "Env:$k" -ErrorAction SilentlyContinue }
  Write-Output "$($r.t) done"
}
foreach ($r in $seq) {
  $f = "$tmp\out_abba_$($r.t)\timing.csv"
  $vals = @()
  foreach ($l in [System.IO.File]::ReadAllLines($f)) {
    $p = $l -split ','
    if ($p.Length -ge 3 -and $p[1] -eq 'laser_match_scan_v3') {
      $v = 0.0; if ([double]::TryParse($p[2], [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$v)) { $vals += $v }
    }
  }
  $s = $vals | Sort-Object; $n = $s.Count
  Write-Output ("{0,-8} n={1}  med={2,6:F2}  p90={3,6:F2}" -f $r.t, $n, $s[[int](($n-1)/2)], $s[[int][Math]::Floor(($n-1)*0.9)])
}
