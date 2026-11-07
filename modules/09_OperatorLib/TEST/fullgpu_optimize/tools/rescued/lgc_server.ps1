$cmd = 'C:\Users\DUUMM\AppData\Local\Temp\opencode\setfreq.txt'
$done = 'C:\Users\DUUMM\AppData\Local\Temp\opencode\lgc_done.txt'
while ($true) {
    if (Test-Path $cmd) {
        $f = (Get-Content $cmd -Raw).Trim()
        Remove-Item $cmd -ErrorAction SilentlyContinue
        if ($f -eq 'EXIT') { nvidia-smi -rgc | Out-Null; 'ok' | Set-Content $done; break }
        nvidia-smi -lgc "$f,$f" | Out-Null
        'ok' | Set-Content $done
    }
    Start-Sleep -Milliseconds 400
}
