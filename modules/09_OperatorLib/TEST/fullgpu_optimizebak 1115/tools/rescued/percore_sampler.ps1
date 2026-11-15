param(
    [string]$out = 'C:\Users\DUUMM\AppData\Local\Temp\opencode\percore_log.csv',
    [int]$secs = 900,
    [int]$interval = 2
)
$sw = [Diagnostics.Stopwatch]::StartNew()
$fs = [System.IO.StreamWriter]::new($out, $false)
$hdr = 'time,tz_temp_c,gpu_temp,sm_mhz,power_w'
$lpCount = 0
$ci0 = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation | Where-Object { $_.Name -ne '_Total' }
$lpCount = @($ci0).Count
for ($i = 0; $i -lt $lpCount; ++$i) { $hdr = $hdr + ',u' + $i + ',p' + $i + ',f' + $i }
$fs.WriteLine($hdr)
while ($sw.Elapsed.TotalSeconds -lt $secs) {
    $tzT = -273.15
    try {
        $tz = Get-CimInstance Win32_PerfFormattedData_Counters_ThermalZoneInformation | Select-Object -First 1
        if ($tz -and $tz.HighPrecisionTemperature -gt 0) { $tzT = [double]$tz.HighPrecisionTemperature / 10.0 - 273.15 }
    } catch {}
    $g = & nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader,nounits 2>$null
    $gp = (($g -join '').Trim() -split ',')
    $line = (Get-Date -Format 'HH:mm:ss') + ',' + [Math]::Round($tzT,1) + ',' + $gp[0].Trim() + ',' + $gp[1].Trim() + ',' + $gp[2].Trim()
    $rows = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation | Where-Object { $_.Name -ne '_Total' } | Sort-Object { [int]($_.Name -split ',')[1] }
    $i = 0
    foreach ($r in $rows) {
        $line = $line + ',' + $r.PercentProcessorTime + ',' + $r.PercentProcessorPerformance + ',' + $r.ProcessorFrequency
        ++$i
    }
    $fs.WriteLine($line)
    $fs.Flush()
    Start-Sleep -Seconds $interval
}
$fs.Close()
