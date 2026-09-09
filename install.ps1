# MoGe-nuke installer (Windows).
#
#   powershell -ExecutionPolicy Bypass -File install.ps1 [-Cuda cu128] [-SkipModel] [-SkipBuild]
#                                                       [-Offline WHEELHOUSE] [-Python path\to\python.exe]
#
# 1. creates .venv with uv (Python 3.12) and installs torch + MoGe + Triton
# 2. downloads the MoGe-3 checkpoint to models\ (about 5 GB, skip with -SkipModel)
# 3. builds the OFX plugin if Visual Studio C++ tools are present, else uses
#    ofx\prebuilt, and installs it where Nuke looks for OFX plugins
# 4. registers nuke\ with Nuke (adds nuke.pluginAddPath to ~\.nuke\init.py)
#
# -Offline WHEELHOUSE installs from a directory made by tools\make_wheelhouse.ps1
# on a connected machine (see docs\OFFLINE.md); the model must already be in
# models\. -Python picks the interpreter for the venv (offline machines).
#
# Re-running is safe; each step skips what is already done.
param(
    [string]$Cuda = "cu128",     # torch wheel index: cu126 / cu128 / cu130 ...
    [switch]$SkipModel,
    [switch]$SkipBuild,
    [switch]$Prebuilt,           # force the prebuilt .ofx even if VS is available
    [string]$OfxDir = "",        # override the OFX plugin directory
    [string]$Offline = "",       # wheelhouse directory for air-gapped installs
    [string]$Python = ""         # interpreter to build the venv from (default: uv-managed 3.12)
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootFwd = $root -replace "\\", "/"
Set-Location $root

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }

# Run a python snippet in the venv and return $true if it exits 0. Native
# stderr must not become a terminating error under $ErrorActionPreference=Stop
# (Windows PowerShell 5.1 does that), hence the dance.
function Probe($code) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $py -c $code *> $null } catch {}
    $ok = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $old
    return $ok
}

# ---------------------------------------------------------------- 1. venv
Step "Python environment"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found. Install it with:  powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`"  then re-run."
}
if (-not $Offline -and -not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git not found (MoGe's dependencies are installed from GitHub). Install Git for Windows and re-run."
}
if ($Offline) { $Offline = (Resolve-Path $Offline).Path }
$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"
$env:VIRTUAL_ENV = $venv
if (-not (Test-Path $py)) {
    $pyArg = if ($Python) { $Python } else { "3.12" }
    if ($Offline) { uv venv --offline --python $pyArg $venv } else { uv venv --python $pyArg $venv }
    if ($LASTEXITCODE) { throw "uv venv failed (offline machines: pass -Python <existing 3.10-3.12 interpreter>)" }
}
if ($Offline) {
    if (-not (Probe "import torch, triton, flex_gemm, moge")) {
        Write-Host "installing from wheelhouse $Offline ..."
        uv pip install --offline --no-index --find-links $Offline -r (Join-Path $Offline "requirements.txt")
        if ($LASTEXITCODE) { throw "wheelhouse install failed" }
        uv pip install --offline --no-index --find-links $Offline --no-deps -e (Join-Path $root "third_party\MoGe")
        if ($LASTEXITCODE) { throw "MoGe install failed" }
    }
} else {
    if (-not (Probe "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)")) {
        Write-Host "installing torch ($Cuda) ..."
        uv pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
        if ($LASTEXITCODE) { throw "torch install failed" }
    }
    if (-not (Probe "import flex_gemm, moge, triton")) {
        Write-Host "installing MoGe + dependencies ..."
        uv pip install -e (Join-Path $root "third_party\MoGe")
        if ($LASTEXITCODE) { throw "MoGe install failed" }
        uv pip install triton-windows safetensors "opencv-python-headless<5"
        if ($LASTEXITCODE) { throw "dependency install failed" }
    }
}
& $py -c "import torch, triton, flex_gemm, moge; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
if ($LASTEXITCODE) { throw "environment check failed" }

# ---------------------------------------------------------------- 2. model
if (-not $SkipModel) {
    Step "Model checkpoint"
    if ($Offline) {
        $have = Get-ChildItem (Join-Path $root "models") -Filter "moge-3-vitg.*" -ErrorAction SilentlyContinue
        if (-not $have) { throw "offline: put the checkpoint in models\ first (see docs\OFFLINE.md)" }
        $have | ForEach-Object { Write-Host "present: $($_.FullName)" }
    } else {
        & $py (Join-Path $root "tools\download_model.py")
        if ($LASTEXITCODE) { throw "model download failed" }
    }
}

# ---------------------------------------------------------------- 3. plugin
Step "OFX plugin"
$bundleSrc = $null
if (-not $SkipBuild -and -not $Prebuilt) {
    try {
        & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "ofx\build.ps1")
        if ($LASTEXITCODE) { throw "build failed" }
        $bundleSrc = Join-Path $root "ofx\build\MoGe3.ofx.bundle"
    } catch {
        Write-Warning "build skipped ($($_.Exception.Message)); using the prebuilt plugin"
    }
}
if (-not $bundleSrc) { $bundleSrc = Join-Path $root "ofx\prebuilt\MoGe3.ofx.bundle" }
if (-not (Test-Path (Join-Path $bundleSrc "Contents\Win64\MoGe3.ofx"))) { throw "no plugin bundle at $bundleSrc" }

# where Nuke looks: OFX_PLUGIN_PATH entries, plus the system-wide default
$dest = $OfxDir
if (-not $dest -and $env:OFX_PLUGIN_PATH) {
    $entries = @($env:OFX_PLUGIN_PATH.Split(";") | Where-Object { $_.Trim() -ne "" })
    if ($entries.Count -gt 0) { $dest = $entries[0].Trim() }
}
$systemDir = Join-Path $env:CommonProgramFiles "OFX\Plugins"
$userDir = Join-Path $env:USERPROFILE ".nuke\OFXPlugins"
$candidates = @()
if ($dest) { $candidates += $dest }
$candidates += $systemDir
$candidates += $userDir
$installed = $null
foreach ($dir in $candidates) {
    try {
        New-Item -ItemType Directory -Force $dir | Out-Null
        $target = Join-Path $dir "MoGe3.ofx.bundle"
        if (Test-Path $target) { Remove-Item -Recurse -Force $target }
        Copy-Item -Recurse $bundleSrc $target
        $installed = $target
        break
    } catch {
        Write-Warning "cannot write to $dir ($($_.Exception.Message))"
    }
}
if (-not $installed) { throw "could not install the plugin anywhere. Close Nuke (it locks the .ofx) or run as administrator." }

# tell the plugin where the repo is
$cfg = @(
    "# written by install.ps1 -- paths the MoGe3 node uses as defaults",
    "root=$rootFwd",
    "python=$rootFwd/.venv/Scripts/python.exe",
    "daemon=$rootFwd/daemon/moge_daemon.py"
)
foreach ($name in @("moge-3-vitg.safetensors", "moge-3-vitg.pt")) {
    if (Test-Path (Join-Path $root "models\$name")) { $cfg += "model=$rootFwd/models/$name"; break }
}
Set-Content -Path (Join-Path $installed "Contents\Win64\moge3.cfg") -Value $cfg -Encoding ASCII
Write-Host "installed: $installed"

# the user directory is only scanned if OFX_PLUGIN_PATH says so
$installedDir = Split-Path -Parent $installed
if ($installedDir -ne $systemDir) {
    $cur = [Environment]::GetEnvironmentVariable("OFX_PLUGIN_PATH", "User")
    $curEntries = @(); if ($cur) { $curEntries = @($cur.Split(";") | ForEach-Object { $_.Trim().TrimEnd("\") }) }
    if ($curEntries -notcontains $installedDir.TrimEnd("\")) {
        $new = if ($cur) { "$cur;$installedDir" } else { $installedDir }
        [Environment]::SetEnvironmentVariable("OFX_PLUGIN_PATH", $new, "User")
        Write-Host "set user OFX_PLUGIN_PATH=$new  (log out / in, or restart Nuke from a new shell)"
    }
}

# ---------------------------------------------------------------- 4. menu
Step "Nuke menu"
$nukeDir = Join-Path $env:USERPROFILE ".nuke"
New-Item -ItemType Directory -Force $nukeDir | Out-Null
$init = Join-Path $nukeDir "init.py"
$line = "nuke.pluginAddPath(`"$rootFwd/nuke`")"
if (-not (Test-Path $init) -or -not (Select-String -Path $init -SimpleMatch $line -Quiet)) {
    Add-Content -Path $init -Value @("", "# MoGe-nuke (MoGe3 OFX node + daemon)", $line)
    Write-Host "added to $init"
} else {
    Write-Host "already registered in $init"
}

Write-Host ""
Write-Host "Done. Restart Nuke, then: Nodes > ML > MoGe3 > Depth + Normals" -ForegroundColor Green
Write-Host "The first render starts the daemon and loads the model (15-60 s)."
