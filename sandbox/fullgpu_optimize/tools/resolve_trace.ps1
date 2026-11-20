# resolve_trace.ps1 - match image-relative offsets from census alloc trace against linker MAP
# Usage: resolve_trace.ps1 -Trace <alloc.log> -Map <speed_replay.map> [-Top 20]
param(
    [Parameter(Mandatory=$true)][string]$Trace,
    [Parameter(Mandatory=$true)][string]$Map,
    [int]$Top = 20
)

# 1) parse map: lines like " 0001:000123ab  SymbolName  004133ab f obj.obj"
$syms = New-Object System.Collections.Generic.List[object]
foreach ($line in [System.IO.File]::ReadAllLines($Map)) {
    if ($line -match '^\s*([0-9a-fA-F]{4}):([0-9a-fA-F]{8})\s+(\S+)\s+[0-9a-fA-F]{8}') {
        $sec = [Convert]::ToInt32($Matches[1],16); $off = [Convert]::ToInt64($Matches[2],16); $nm = $Matches[3]
        if ($sec -eq 1) { $rva = 0x1000 + $off; $syms.Add([PSCustomObject]@{rva=$rva; name=$nm}) }
    }
}
$sorted = $syms | Sort-Object rva
$ras = $sorted | ForEach-Object { $_.rva }
"map symbols(section1): $($sorted.Count)"

# 2) walk trace: for each [cudaMalloc] event, resolve its exe frames to nearest symbol
$events = New-Object System.Collections.Generic.List[object]
$curFrames = New-Object System.Collections.Generic.List[long]
function Resolve([long]$rva) {
    $idx = [Array]::BinarySearch([long[]]$script:ras, $rva)
    if ($idx -ge 0) { return $script:sorted[$idx].name }
    $idx = -($idx + 1) - 1
    if ($idx -lt 0) { return "<below-first>" }
    return $script:sorted[$idx].name
}
foreach ($line in [System.IO.File]::ReadAllLines($Trace)) {
    if ($line -match '^\[cudaMalloc\]') {
        if ($curFrames.Count -gt 0) {
            $chain = ($curFrames | ForEach-Object { Resolve $_ }) -join ' <- '
            $events.Add($chain)
        }
        $curFrames.Clear()
    } elseif ($line -match 'speed_replay\.exe!\+0x([0-9a-fA-F]+)') {
        $curFrames.Add([Convert]::ToInt64($Matches[1],16))
    }
}
if ($curFrames.Count -gt 0) {
    $chain = ($curFrames | ForEach-Object { Resolve $_ }) -join ' <- '
    $events.Add($chain)
}
"resolved events: $($events.Count)"
"=== 业务调用链 TOP$Top（浅帧在右，深帧在左；取每事件最浅 3 帧归类）==="
$events | ForEach-Object {
    $f = $_ -split ' <- '
    ($f | Select-Object -Last 3) -join ' <- '
} | Group-Object | Sort-Object Count -Descending | Select-Object -First $Top | Format-Table Count,Name -AutoSize -Wrap
