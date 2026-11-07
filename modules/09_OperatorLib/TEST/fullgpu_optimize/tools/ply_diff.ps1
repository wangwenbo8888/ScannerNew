# ply_diff.ps1 - bidirectional nearest-neighbor PLY comparison (grid hash, order-independent)
# Usage: ply_diff.ps1 -A <a.ply> -B <b.ply> [-Tol 2.0] [-MaxMiss 2] [-Stride 1]
# Verdict: missAB/missBA each <= MaxMiss and |nA-nB|<=2 -> PASS
param(
    [Parameter(Mandatory=$true)][string]$A,
    [Parameter(Mandatory=$true)][string]$B,
    [double]$Tol = 2.0,
    [int]$MaxMiss = 2,
    [int]$Stride = 1
)

function Read-PlyXyz([string]$path, [int]$stride) {
    $pts = New-Object System.Collections.Generic.List[object]
    $inBody = $false
    $n = 0
    foreach ($line in [System.IO.File]::ReadLines($path)) {
        if (-not $inBody) {
            if ($line -match '^end_header') { $inBody = $true }
            continue
        }
        $f = $line.Trim() -split '\s+'
        if ($f.Count -lt 3) { continue }
        if (($n % [Math]::Max(1,$stride)) -eq 0) {
            $pts.Add(@([double]$f[0], [double]$f[1], [double]$f[2]))
        }
        $n++
    }
    return $pts
}

function Nearest-Miss($from, $to, [double]$tol) {
    $grid = @{}
    for ($j = 0; $j -lt $to.Count; $j++) {
        $p = $to[$j]
        $key = "{0}:{1}:{2}" -f [Math]::Floor($p[0]/$tol), [Math]::Floor($p[1]/$tol), [Math]::Floor($p[2]/$tol)
        if (-not $grid.ContainsKey($key)) { $grid[$key] = New-Object System.Collections.Generic.List[int] }
        $grid[$key].Add($j)
    }
    $miss = 0; $maxD = 0.0
    $missSamples = New-Object System.Collections.Generic.List[string]
    for ($i = 0; $i -lt $from.Count; $i++) {
        $p = $from[$i]
        $cx = [Math]::Floor($p[0]/$tol); $cy = [Math]::Floor($p[1]/$tol); $cz = [Math]::Floor($p[2]/$tol)
        $best = [double]::MaxValue; $found = $false
        foreach ($dx in @(-1,0,1)) { foreach ($dy in @(-1,0,1)) { foreach ($dz in @(-1,0,1)) {
            $key = "{0}:{1}:{2}" -f ($cx+$dx), ($cy+$dy), ($cz+$dz)
            if (-not $grid.ContainsKey($key)) { continue }
            foreach ($j in $grid[$key]) {
                $q = $to[$j]
                $d = [Math]::Abs($p[0]-$q[0]) + [Math]::Abs($p[1]-$q[1]) + [Math]::Abs($p[2]-$q[2])
                if ($d -lt $best) { $best = $d }
            }
        }}}
        if ($best -le 3.0*$tol) { $found = $true; if ($best -gt $maxD) { $maxD = $best } }
        if (-not $found) {
            $miss++
            if ($missSamples.Count -lt 20) { $missSamples.Add(("  miss[{0}]=({1:F2},{2:F2},{3:F2})" -f $i,$p[0],$p[1],$p[2])) }
        }
    }
    return @{ miss = $miss; maxD = $maxD; samples = $missSamples }
}

$pa = Read-PlyXyz $A $Stride
$pb = Read-PlyXyz $B $Stride
if ($pa.Count -eq 0 -or $pb.Count -eq 0) {
    Write-Output "[ply] EMPTY (nA=$($pa.Count) nB=$($pb.Count)) -> FAIL"; exit 1
}
$numDiff = [Math]::Abs($pa.Count - $pb.Count)
$ab = Nearest-Miss $pa $pb $Tol
$ba = Nearest-Miss $pb $pa $Tol
$verdict = if ($ab.miss -le $MaxMiss -and $ba.miss -le $MaxMiss -and $numDiff -le 2) { "PASS" } else { "FAIL" }
Write-Output ("[ply] nA={0} nB={1} stride={2} missAB={3} missBA={4} numDiff={5} maxD={6:F2} -> {7}" -f `
    $pa.Count, $pb.Count, $Stride, $ab.miss, $ba.miss, $numDiff, [Math]::Max($ab.maxD,$ba.maxD), $verdict)
$ab.samples | ForEach-Object { Write-Output $_ }
$ba.samples | ForEach-Object { Write-Output $_ }
exit $(if ($verdict -eq "PASS") { 0 } else { 1 })
