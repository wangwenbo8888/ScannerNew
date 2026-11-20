# compare_run.ps1 - per-frame report.csv comparison (mirrors main.cpp CompareRuns L261-283)
# Column indices (0-based): 1=markersDetected 2=markersValid 3=regSource 4=regMatched
#   5=dt_mm 6=drot_deg 7=laserMatched 8=laserPoints3D 9=rowResidMedPx 10=dispMedPx
# Extras vs C++ authority: jitter budget (<=1% frames) for RANSAC micro-metrics
#   (regMatched/dt/drot) per README 8.2 jitter doctrine; documented here.
param(
    [Parameter(Mandatory=$true)][string]$Golden,
    [Parameter(Mandatory=$true)][string]$Candidate,
    [switch]$VerboseDiff
)

function Read-Report([string]$path) {
    $map = @{}
    foreach ($line in [System.IO.File]::ReadAllLines($path)) {
        if ($line -match '^cycle,') { continue }
        $f = $line -split ','
        if ($f.Count -lt 11) { continue }
        if (-not $map.ContainsKey($f[0])) { $map[$f[0]] = $f }
    }
    return $map
}

$a = Read-Report $Golden
$b = Read-Report $Candidate
if ($a.Count -eq 0 -or $b.Count -eq 0) {
    Write-Output "[compare] EMPTY report (A=$($a.Count) B=$($b.Count)) -> FAIL"
    exit 1
}

$bad = 0; $checked = 0; $flips = 0; $inFlip = $false; $jitter = 0
$flipBudget = [Math]::Max(1, [Math]::Floor($a.Count / 100))
$diffLines = New-Object System.Collections.Generic.List[string]

foreach ($k in $a.Keys) {
    if (-not $b.ContainsKey($k)) {
        $bad++; $diffLines.Add("[compare] cycle $k missing in B"); continue
    }
    $x = $a[$k]; $y = $b[$k]; $checked++
    if ($x[3] -ne $y[3]) {
        $flips++
        if ($flips -gt $flipBudget) { $bad++; $diffLines.Add("[compare] cycle $k regSource $($x[3]) vs $($y[3])") }
        $inFlip = $true
    } else { $inFlip = $false }

    $fail = { param($nm, $va, $vb) $script:bad++; $script:diffLines.Add("[compare] cycle $k $nm $va vs $vb") }
    $jfail = { param($nm, $va, $vb)
        $script:jitter++
        if ($script:jitter -gt $script:flipBudget) { & $fail $nm $va $vb } }

    if ([Math]::Abs([double]$x[1]-[double]$y[1]) -gt 2) { & $fail 'markersDetected' $x[1] $y[1] }
    if ([Math]::Abs([double]$x[2]-[double]$y[2]) -gt 1) { & $fail 'markersValid' $x[2] $y[2] }
    if (-not $inFlip) {
        if ([Math]::Abs([double]$x[4]-[double]$y[4]) -gt 2) { & $jfail 'regMatched' $x[4] $y[4] }
        $da=[double]$x[5]; $db=[double]$y[5]
        if ([Math]::Abs($da-$db) -gt [Math]::Max(2.5, 0.2*[Math]::Abs($da))) { & $jfail 'dt_mm' $x[5] $y[5] }
        $da=[double]$x[6]; $db=[double]$y[6]
        if ([Math]::Abs($da-$db) -gt [Math]::Max(0.4, 0.1*[Math]::Abs($da))) { & $jfail 'drot_deg' $x[6] $y[6] }
        foreach ($i in @(7,8)) {
            $da=[double]$x[$i]; $db=[double]$y[$i]
            if ([Math]::Abs($da-$db) -gt [Math]::Max(4.0, 0.01*[Math]::Abs($da))) { & $fail "col$i" $x[$i] $y[$i] }
        }
        if ([Math]::Abs([double]$x[9]-[double]$y[9]) -gt 0.01) { & $fail 'rowResid' $x[9] $y[9] }
        if ([Math]::Abs([double]$x[10]-[double]$y[10]) -gt 0.1) { & $fail 'dispMed' $x[10] $y[10] }
    }
}

if ($VerboseDiff) { $diffLines | Select-Object -First 40 | ForEach-Object { Write-Output $_ } }
$verdict = if ($bad -eq 0) { "PASS" } else { "FAIL" }
Write-Output "[compare] checked=$checked diffs=$bad regSrcFlips=$flips jitterBudgetUsed=$jitter/$flipBudget -> $verdict"
exit $(if ($bad -eq 0) { 0 } else { 1 })
