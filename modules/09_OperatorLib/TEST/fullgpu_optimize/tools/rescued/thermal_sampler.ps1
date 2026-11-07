param(
    [string]$out = 'C:\Users\DUUMM\AppData\Local\Temp\opencode\thermal_log.csv',
    [int]$secs = 900,
    [int]$interval = 2
)
$sw = [Diagnostics.Stopwatch]::StartNew()
$fs = [System.IO.StreamWriter]::new($out, $false)
$fs.WriteLine('time,cpu_pct,cpu_mhz,tz_temp_c,tz_passive_pct,tz_throttle,gpu_temp,gpu_util,sm_mhz,mem_mb,power_w,throttle_bits')
while ($sw.Elapsed.TotalSeconds -lt $secs) {
    $cpu = -1; $mhz = -1
    try {
        $ci = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation -Filter "Name='_Total'"
        if ($ci) { $cpu = [double]$ci.PercentProcessorPerformance; $mhz = [double]$ci.ProcessorFrequency }
    } catch {}
    $tzT = -273.15; $tzP = -1; $tzR = -1
    try {
        $tz = Get-CimInstance Win32_PerfFormattedData_Counters_ThermalZoneInformation | Select-Object -First 1
        if ($tz) {
            if ($tz.HighPrecisionTemperature -gt 0) { $tzT = [double]$tz.HighPrecisionTemperature / 10.0 - 273.15 }
            elseif ($tz.Temperature -gt 0) { $tzT = [double]$tz.Temperature - 273.15 }
            $tzP = [double]$tz.PercentPassiveLimit
            $tzR = [double]$tz.ThrottleReasons
        }
    } catch {}
    $g = & nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,clocks.sm,memory.used,power.draw,clocks_event_reasons.active --format=csv,noheader,nounits 2>$null
    $line = ($g -join '').Trim()
    $fs.WriteLine((Get-Date -Format 'HH:mm:ss') + ',' + $cpu + ',' + $mhz + ',' + [Math]::Round($tzT,1) + ',' + $tzP + ',' + $tzR + ',' + $line)
    $fs.Flush()
    Start-Sleep -Seconds $interval
}
$fs.Close()
