param(
    [int]$Lp = 10,
    [int]$DelaySec = 40,
    [int]$DurSec = 15
)
Start-Sleep -Seconds $DelaySec
$ok = $false
try {
    $p = Get-Process -Id $PID
    if ($Lp -lt [Environment]::ProcessorCount) {
        $mask = [IntPtr][long](1L -shl $Lp)
        $p.ProcessorAffinity = $mask
        $ok = $true
    }
} catch {}
$end = (Get-Date).AddSeconds($DurSec)
$x = 1.1
while ((Get-Date) -lt $end) { $x = $x * 1.0000001; if ($x -gt 1e30) { $x = 1.1 } }
"lp=$Lp pinned=$ok" | Set-Content ('C:\Users\DUUMM\AppData\Local\Temp\opencode\probe_' + $Lp + '_' + $DelaySec + '.txt')
