# run_abba.ps1 - formal back-to-back A/B speed verification (archive discipline: paired diffs only)
# Usage: run_abba.ps1 -ExeA <old.exe> -ExeB <new.exe> [-Repeat 3] [-Config <json>] [-DataDir <dir>]
param(
    [Parameter(Mandatory=$true)][string]$ExeA,
    [Parameter(Mandatory=$true)][string]$ExeB,
    [int]$Repeat = 3,
    [string]$Config = "E:\JEAMMWARE2601001\modules\09_operatorlib\TEST\fullgpu_optimize\configs\config_baseline.json",
    [string]$DataDir = "E:\JEAMMWARE2601001\modules\09_operatorlib\TEST\fullgpu_optimize\date_in\laser-mark1688-1006\ALL_raw",
    [string]$OutRoot = "C:\Users\DUUMM\AppData\Local\Temp\opencode\abba",
    [hashtable]$ArmBEnv = $null,   # arm-B-only env injection (e.g. @{ 'JMW_V3FRONTGRAPH' = '1' }; arm A never gets these)
    [hashtable]$BaseEnv = $null    # common env for BOTH arms (e.g. @{ 'JMW_1BGRAPH' = '1' })
)
$env:PATH = 'C:\opencv-cuda-4.13.0\x64\vc17\bin;' + $env:PATH
New-Item -ItemType Directory -Force $OutRoot | Out-Null

# interleaved order A-B-A-B-B-A-B-A (drift-balanced, 4 pairs)
# 2026-10-27 tool fixes: (1) wipe per-arm dump dir before run - stale files from
#   earlier sessions caused false median parses (crash-before-write read old csv);
# (2) retry up to 3x when a run aborts without producing fps_summary (baseline
#   binaries predating the 1a capture fix abort ~25%/run under back-to-back load);
# (3) pairing keys fixed to global-seq keys actually produced (A1,B2,A3,B4,B5,A6,B7,A8)
$order = @('A','B','A','B','B','A','B','A')
$results = @{}   # arm+n -> @{ median; steady }
$seq = 1
foreach ($arm in $order) {
    $exe = if ($arm -eq 'A') { $ExeA } else { $ExeB }
    $dir = Join-Path $OutRoot ("{0}{1}" -f $arm, $seq)
    # env hygiene: only JMW_FIVECHAIN set; clear experiment vars
    # [batch H] expanded: legacy six + new graph-family gates (FLATGRAPH/RLEGRAPH/
    # FUSEGRAPH/1BBIGGRAPH) + master switch NOGRAPH - stale residues silently
    # pin an arm to a non-default graph stack (red-team A S1-2)
    Remove-Item Env:JMW_FC_FULLGPU, Env:JMW_CENSUS, Env:JMW_CENSUS_TRACE, Env:JMW_CENSUS_OUT, Env:JMW_1BGRAPH, Env:JMW_V3FRONTGRAPH, Env:JMW_FLATGRAPH, Env:JMW_RLEGRAPH, Env:JMW_FUSEGRAPH, Env:JMW_1BBIGGRAPH, Env:JMW_NOGRAPH -ErrorAction SilentlyContinue
    $env:JMW_FIVECHAIN = '1'
    if ($BaseEnv) { foreach ($k in $BaseEnv.Keys) { Set-Item "Env:$k" $BaseEnv[$k] } }
    if ($arm -eq 'B' -and $ArmBEnv) {
        foreach ($k in $ArmBEnv.Keys) { Set-Item "Env:$k" $ArmBEnv[$k] }
    }
    $med = -1.0; $steady = -1.0; $exit = 0
    for ($try = 1; $try -le 3; $try++) {
        if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
        New-Item -ItemType Directory -Force $dir | Out-Null
        # stderr to disk per run (crash forensics - never swallow with *> $null)
        & $exe --config $Config --data-dir $DataDir --cycles 0 --repeat $Repeat --lean --quiet --dump-dir $dir > (Join-Path $dir 'run.out.log') 2> (Join-Path $dir 'run.err.log')
        $exit = $LASTEXITCODE
        # arm-B env must not leak into arm A - strip immediately after run
        if ($ArmBEnv) { foreach ($k in $ArmBEnv.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue } }
        # parse fps_summary ALL median + fc_summary steady
        $fs = Join-Path $dir 'fps_summary.csv'
        if (Test-Path $fs) {
            foreach ($l in Get-Content $fs) {
                if ($l -match '^ALL,(\d+),([\d.]+)') { $med = [double]$Matches[2] }
            }
        }
        if ($med -ge 0) { break }   # got valid summary; else crash-before-write -> retry
        Write-Output ("arm {0}{1}: attempt {2} aborted (exit={3}), no summary - retry" -f $arm, $seq, $try, $exit)
    }
    $fc = Join-Path $dir 'fc_summary.csv'
    if (Test-Path $fc) {
        foreach ($l in Get-Content $fc) {
            if ($l -match '^steady_fps_median,([\d.]+)') { $steady = [double]$Matches[1] }
        }
    }
    $t = (Get-Item $exe).LastWriteTime.ToString('HH:mm')
    Write-Output ("arm {0}{1}: exit={2} median={3:F1} steady={4:F1}" -f $arm, $seq, $exit, $med, $steady)
    $results[("{0}{1}" -f $arm, $seq)] = @{ med = $med; steady = $steady }
    $seq++
}
Remove-Item Env:JMW_FIVECHAIN -ErrorAction SilentlyContinue
if ($BaseEnv) { foreach ($k in $BaseEnv.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue } }

# pairs by adjacency of actual run sequence A1,B2,A3,B4,B5,A6,B7,A8;
# d expresses which subtraction yields B-minus-A
$pairs = @(
    @{ a = 'A1'; b = 'B2'; d = 'B-A' }, @{ a = 'A3'; b = 'B4'; d = 'B-A' },
    @{ a = 'B5'; b = 'A6'; d = 'A-B' }, @{ a = 'B7'; b = 'A8'; d = 'A-B' }
)
$diffs = New-Object System.Collections.Generic.List[double]
Write-Output "`n=== paired diffs (B=new minus A=old, FPS median) ==="
foreach ($p in $pairs) {
    $ra = $results[$p.a]; $rb = $results[$p.b]
    if (-not $ra -or -not $rb) { continue }
    if ($ra.med -lt 0 -or $rb.med -lt 0) {
        Write-Output ("pair {0} vs {1}: SKIPPED (invalid arm, aborted all retries)" -f $p.a, $p.b)
        continue
    }
    $d = if ($p.d -eq 'B-A') { $rb.med - $ra.med } else { $ra.med - $rb.med }
    $diffs.Add($d)
    Write-Output ("pair {0} vs {1} ({2}): dFps={3:+0.0;-0.0}" -f $p.a, $p.b, $p.d, $d)
}
if ($diffs.Count -ge 3) {
    $srt = $diffs | Sort-Object
    $median = $srt[[int](($srt.Count - 1) / 2)]
    $pos = @($diffs | Where-Object { $_ -ge -3 }).Count
    Write-Output ("`nmedian(dFps)={0:+0.00;-0.00}  pairs(dFps>=-3)={1}/{2}" -f $median, $pos, $diffs.Count)
    if ($median -ge -3 -and $pos -ge 3) {
        if ($median -ge 3) { Write-Output "VERDICT: IMPROVED (median >= +3, no-regression met)" }
        else { Write-Output "VERDICT: NO-REGRESSION (B not worse; gain within noise band)" }
    } else {
        Write-Output "VERDICT: REGRESSION SUSPECT (needs investigation)"
    }
}
