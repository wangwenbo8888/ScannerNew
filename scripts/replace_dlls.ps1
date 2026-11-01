# ============================================================================
# replace_dlls.ps1 — 07/09 算法 DLL 快速替换流（261031 D1-lite）
#
# 用法（仓库根）：
#   powershell -File scripts/replace_dlls.ps1                     # 默认 Release，编两库即生效
#   powershell -File scripts/replace_dlls.ps1 -Only op            # 只更新 OperatorLib.dll
#   powershell -File scripts/replace_dlls.ps1 -Only pm            # 只更新 PipelineManager.dll
#   powershell -File scripts/replace_dlls.ps1 -Dest D:\delivery   # 拷到交付目录
#   powershell -File scripts/replace_dlls.ps1 -SkipBuild          # 只拷贝不构建
#
# 安全护栏（内置，无需人工）：
#   · 契约指纹＝公共头闭包＋工具链串的 SHA-256，构建期内嵌 DLL
#   · exe 启动校验 07；07 会话配置校验 09——指纹不一致直接拒绝
#   · 只改实现（头不动）→ 指纹不变 → 换 DLL 即算法热更 ✓
#   · 改公共头/换 OpenCV/换工具链 → 指纹变 → 拒绝并要求全量重编 ✓
# ============================================================================
param(
    [ValidateSet('Release', 'Debug')][string]$Config = 'Release',
    [ValidateSet('op', 'pm', 'both')][string]$Only = 'both',
    [string]$Dest = '',
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent

if (-not $SkipBuild) {
    $targets = @()
    if ($Only -in 'op', 'both') { $targets += 'OperatorLib' }
    if ($Only -in 'pm', 'both') { $targets += 'PipelineManager' }
    Write-Host "== 增量构建 [$($targets -join ' + ')] ($Config) =="
    & cmake --build "$root/build" --config $Config --target @targets
    if ($LASTEXITCODE -ne 0) { throw "构建失败（exit $LASTEXITCODE）" }
}

$bin = Join-Path $root "build/bin/$Config"
$dst = if ($Dest) { $Dest } else { $bin }
if (-not (Test-Path $dst)) { New-Item -ItemType Directory $dst -Force | Out-Null }

$map = @{ op = 'OperatorLib.dll'; pm = 'PipelineManager.dll' }
foreach ($k in @('op', 'pm')) {
    if ($Only -ne $k -and $Only -ne 'both') { continue }
    $f = $map[$k]
    $src = Join-Path $bin $f
    if (-not (Test-Path $src)) { throw "找不到 $src（先构建）" }
    Copy-Item $src (Join-Path $dst $f) -Force
    Write-Host "已替换: $dst\$f"
}

Write-Host @"
== 完成 ==
替换后首启会做契约校验（日志 [契约] 行）：
  · 指纹一致 → 算法更新生效
  · 指纹不一致 → 拒绝启动，说明公共头已变，需全量重编（勿强行绕过）
提示：替换前请退出 LeadScannerK2.exe（运行中 DLL 被锁）。
"@
