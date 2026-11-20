# deep_chains.ps1 - full call-chain resolution of census alloc trace against linker MAP
param(
    [Parameter(Mandatory=$true)][string]$Trace,
    [Parameter(Mandatory=$true)][string]$Map,
    [string]$Filter = '',
    [int]$Top = 10
)
$syms = New-Object System.Collections.Generic.List[object]
foreach ($line in [System.IO.File]::ReadAllLines($Map)) {
    if ($line -match '^\s*0001:([0-9a-fA-F]{8})\s+(\S+)\s+[0-9a-fA-F]{8}') {
        $syms.Add([PSCustomObject]@{rva = 0x1000 + [Convert]::ToInt64($Matches[1],16); name = $Matches[2]})
    }
}
$sorted = @($syms | Sort-Object rva)
$script:ras = [long[]]($sorted | ForEach-Object { $_.rva })
function ResolveSym([long]$rva) {
    $i = [Array]::BinarySearch($script:ras, $rva)
    if ($i -ge 0) { return $sorted[$i].name }
    $i = -($i + 1) - 1
    if ($i -lt 0) { return '<start>' }
    return $sorted[$i].name
}
$chains = @{}
$order = New-Object System.Collections.Generic.List[string]
$cur = New-Object System.Collections.Generic.List[long]
foreach ($line in [System.IO.File]::ReadAllLines($Trace)) {
    if ($line -match '^\[cudaMalloc\]') {
        if ($cur.Count -gt 0) {
            $names = @($cur | ForEach-Object { ResolveSym $_ })
            $k = $names -join ' <- '
            if (-not $chains.ContainsKey($k)) { $chains[$k] = 0; $order.Add($k) }
            $chains[$k]++
        }
        $cur.Clear()
    } elseif ($line -match 'speed_replay\.exe!\+0x([0-9a-fA-F]+)') {
        $cur.Add([Convert]::ToInt64($Matches[1],16))
    }
}
"total events: $(($chains.Values | Measure-Object -Sum).Sum)  distinct chains: $($chains.Count)"
$demote = { param($s) $s -replace '\?([A-Za-z0-9_]+)@[^ ]+','$1()' }
$items = if ($Filter) { $order | Where-Object { $_ -match $Filter } } else { $order }
$items | ForEach-Object {
    [PSCustomObject]@{ Count = $chains[$_]; Chain = (& $demote $_) }
} | Sort-Object Count -Descending | Select-Object -First $Top | Format-Table -AutoSize -Wrap
