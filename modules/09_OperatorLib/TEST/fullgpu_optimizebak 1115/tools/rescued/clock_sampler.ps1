param(
    [string]$out = 'C:\Users\DUUMM\AppData\Local\Temp\opencode\clock_log.csv',
    [int]$secs = 900,
    [int]$interval = 2
)
$sw = [Diagnostics.Stopwatch]::StartNew()
$fs = [System.IO.StreamWriter]::new($out, $false)
$fs.WriteLine('time,cpu_pct,gpu_temp,gpu_util,sm_mhz,mem_mb,power_w,throttle_bits')
while ($sw.Elapsed.TotalSeconds -lt $secs) {
    $cpu = -1
    try {
        $cpu = (Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation -Filter "Name='_Total'").PercentProcessorPerformance
    } catch {}
    $g = & nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,clocks.sm,memory.used,power.draw,clocks_event_reasons.active --format=csv,noheader,nounits 2>$null
    $line = ($g -join '').Trim()
    $fs.WriteLine((Get-Date -Format 'HH:mm:ss') + ',' + $cpu + ',' + $line)
    $fs.Flush()
    Start-Sleep -Seconds $interval
}
$fs.Close()
