<#
.SYNOPSIS
    One-command setup and study run for a Windows CUDA workstation.

.DESCRIPTION
    Does everything WORKSTATION.md describes, in order: checks prerequisites,
    builds the evaluation environment, bootstraps the two isolated matcher
    environments at their pinned commits and checksum-verified weights, records
    this machine's resolved provenance, probes both matchers, and runs the
    study unattended.

    Two things are deliberately not automated away.

    The provenance gate compares exact strings, and a Windows CUDA box resolves
    a different CPython patch release and a CUDA torch build than the recorded
    macOS pins. -AcceptProvenance records what this machine resolved and appends
    a protocol deviation; without it the script stops and shows you the values.
    Either way the re-pin refuses if an upstream commit or a checkpoint hash
    differs, because that means the sources or weights are wrong rather than the
    pins.

    The confirmatory half of the study is gated on a reviewed G1 freeze. Pass
    -SignedOffBy and -ReviewNote to run it; omit them and the run stops after
    the M2 feasibility screen, which is the right choice if you want to read the
    screen before committing to a direction.

.EXAMPLE
    .\scripts\run_study.ps1 -AcceptProvenance -SignedOffBy "Felix Bergmann" `
        -ReviewNote "M2 recommendation reviewed on the workstation."

.EXAMPLE
    .\scripts\run_study.ps1 -SetupOnly
#>

[CmdletBinding()]
param(
    # Study configuration. The historical default keeps the pilot reproducible.
    [string]$Config = "configs/pilot.yaml",
    # Interpreter used to build every environment. Must be CPython 3.11.
    [string]$Python = "py -3.11",
    # Record this machine's resolved matcher provenance in the configuration.
    [switch]$AcceptProvenance,
    # Install CPU-only torch instead of the CUDA build.
    [switch]$Cpu,
    # Stop after the probes; do not start the study.
    [switch]$SetupOnly,
    # Reviewer for the G1 freeze. Without this and -ReviewNote the run stops
    # after the M2 feasibility screen.
    [string]$SignedOffBy = "",
    [string]$ReviewNote = "",
    # Freeze a direction the M2 screen blocks, recording that decision.
    [switch]$AcknowledgeInfeasible,
    # Complete refit resamples for the confirmatory claim (0 = configured value).
    [int]$RefitBootstrap = 0,
    # Print the resolved plan and a wall-time projection, then exit.
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$TorchIndex = if ($Cpu) {
    "https://download.pytorch.org/whl/cpu"
} else {
    # On Windows the default PyPI wheel is CPU-only; CUDA needs this index.
    "https://download.pytorch.org/whl/cu121"
}

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "== $Message" -ForegroundColor Cyan
}

function Invoke-Checked([string]$Description, [scriptblock]$Action) {
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Resolve-Python([string]$Spec) {
    $parts = $Spec.Split(" ", [StringSplitOptions]::RemoveEmptyEntries)
    $exe = $parts[0]
    $rest = if ($parts.Length -gt 1) { $parts[1..($parts.Length - 1)] } else { @() }
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
        throw "Python launcher '$exe' was not found. Install CPython 3.11 and rerun, or pass -Python 'C:\path\to\python.exe'."
    }
    $version = & $exe @rest "-c" "import sys; print('%d.%d' % sys.version_info[:2])"
    if ($LASTEXITCODE -ne 0) { throw "Could not query the interpreter '$Spec'." }
    if ($version.Trim() -ne "3.11") {
        throw "The pinned matcher stack requires CPython 3.11; '$Spec' reports $version. Install python 3.11 and pass -Python."
    }
    return @{ Exe = $exe; Args = $rest }
}

function New-Venv([hashtable]$Interpreter, [string]$Path) {
    if (Test-Path (Join-Path $Path "Scripts\python.exe")) { return }
    # Splatting needs a bare variable: @($Interpreter.Args) would pass the array
    # as one argument rather than expanding it.
    $launcherArgs = $Interpreter.Args
    Invoke-Checked "creating the virtual environment at $Path" {
        & $Interpreter.Exe @launcherArgs "-m" "venv" $Path
    }
}

function Get-VenvPython([string]$Path) {
    return (Join-Path $ProjectRoot (Join-Path $Path "Scripts\python.exe"))
}

function Sync-Repository([string]$Url, [string]$Destination, [string]$Commit) {
    if (-not (Test-Path (Join-Path $Destination ".git"))) {
        Invoke-Checked "cloning $Url" { git clone $Url $Destination }
        Invoke-Checked "checking out $Commit" { git -C $Destination checkout --detach $Commit }
    }
    $head = (git -C $Destination rev-parse HEAD).Trim()
    if ($head -ne $Commit) {
        throw "Refusing existing checkout $Destination : expected $Commit, found $head. Move it aside and rerun."
    }
    git -C $Destination diff --quiet
    if ($LASTEXITCODE -ne 0) {
        throw "Refusing modified upstream checkout: $Destination"
    }
}

function Get-Checkpoint([string]$Url, [string]$Destination, [string]$Sha256) {
    $directory = Split-Path -Parent $Destination
    if (-not (Test-Path $directory)) { New-Item -ItemType Directory -Path $directory -Force | Out-Null }
    if (-not (Test-Path $Destination)) {
        $partial = "$Destination.partial"
        Write-Host "  downloading $(Split-Path -Leaf $Destination)"
        Invoke-WebRequest -Uri $Url -OutFile $partial -UseBasicParsing
        Move-Item -Force $partial $Destination
    }
    $actual = (Get-FileHash -Algorithm SHA256 -Path $Destination).Hash.ToLower()
    if ($actual -ne $Sha256.ToLower()) {
        Remove-Item -Force $Destination
        throw "Checksum mismatch for $Destination : expected $Sha256, got $actual. The file was removed; rerun to fetch it again."
    }
}

# --------------------------------------------------------------------------
Write-Step "Checking prerequisites"

foreach ($tool in @("git")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "'$tool' was not found on PATH. Install it and rerun."
    }
}
if (-not (Get-Command "7z" -ErrorAction SilentlyContinue) -and
    -not (Get-Command "7zz" -ErrorAction SilentlyContinue)) {
    throw "7-Zip was not found on PATH. FIRE ships as a .7z archive and preparation refuses without it. Install 7-Zip (https://www.7-zip.org) and add its folder to PATH."
}
$interpreter = Resolve-Python $Python
Write-Host "  python: $Python (3.11), git, 7-Zip present"
if (-not $Cpu) {
    if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
        $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1)
        Write-Host "  gpu: $gpu"
    } else {
        Write-Warning "nvidia-smi was not found. Installing the CUDA build anyway; pass -Cpu if this machine has no NVIDIA GPU."
    }
}

Write-Step "Building the evaluation environment"
New-Venv $interpreter ".venv"
$Eval = Get-VenvPython ".venv"
Invoke-Checked "upgrading pip" { & $Eval -m pip install --upgrade pip --quiet }
Invoke-Checked "installing warpaudit" {
    & $Eval -m pip install --quiet -c requirements-eval.lock -e ".[dev,report,gbm]"
}
Invoke-Checked "validating the configuration" {
    & $Eval -m warpaudit validate-config --config $Config
}

Write-Step "Fetching pinned matcher sources and checkpoints"
foreach ($directory in @(".pipeline-envs", "vendor", "checkpoints")) {
    if (-not (Test-Path $directory)) { New-Item -ItemType Directory -Path $directory | Out-Null }
}
Sync-Repository "https://github.com/verlab/accelerated_features.git" `
    "vendor/accelerated_features" "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"
Sync-Repository "https://github.com/cvg/LightGlue.git" `
    "vendor/LightGlue" "eb42fee2d71449efb0aa5c10549752b5d75384d8"
Get-Checkpoint "https://raw.githubusercontent.com/verlab/accelerated_features/e92685f57f8318b18725c5c8c0bd28c7fe188d9a/weights/xfeat.pt" `
    "checkpoints/xfeat/xfeat.pt" `
    "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b"
Get-Checkpoint "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth" `
    "checkpoints/lightglue/hub/checkpoints/superpoint_v1.pth" `
    "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e"
Get-Checkpoint "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth" `
    "checkpoints/lightglue/hub/checkpoints/superpoint_lightglue_v0-1_arxiv.pth" `
    "6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a"

Write-Step "Building the isolated matcher environments"
New-Venv $interpreter ".pipeline-envs\xfeat"
$XFeat = Get-VenvPython ".pipeline-envs\xfeat"
Invoke-Checked "upgrading pip in the xfeat environment" {
    & $XFeat -m pip install --upgrade pip --quiet
}
# torch first, from the chosen index: on Windows the default PyPI wheel is
# CPU-only, so a plain requirements install would silently ignore the GPU. The
# local version `2.3.1+cu121` satisfies the lock's `torch==2.3.1`, so the lock
# install that follows keeps it rather than replacing it.
Invoke-Checked "installing torch for xfeat" {
    & $XFeat -m pip install --quiet --index-url $TorchIndex "torch==2.3.1"
}
Invoke-Checked "installing the xfeat lock" {
    & $XFeat -m pip install --quiet --requirement requirements-xfeat.lock
}
& $XFeat -m pip freeze > ".pipeline-envs\xfeat.freeze.txt"

New-Venv $interpreter ".pipeline-envs\sp_lightglue"
$SpLg = Get-VenvPython ".pipeline-envs\sp_lightglue"
Invoke-Checked "upgrading pip in the sp_lightglue environment" {
    & $SpLg -m pip install --upgrade pip --quiet
}
Invoke-Checked "installing torch for sp_lightglue" {
    & $SpLg -m pip install --quiet --index-url $TorchIndex "torch==2.3.1" "torchvision==0.18.1"
}
Invoke-Checked "installing the sp_lightglue lock" {
    & $SpLg -m pip install --quiet --requirement requirements-sp-lightglue.lock
}
# --no-deps: LightGlue's own metadata would pull a second OpenCV and move the
# runtime off the pinned version. The lock stays authoritative.
Invoke-Checked "installing LightGlue from the pinned checkout" {
    & $SpLg -m pip install --quiet --no-deps --editable vendor/LightGlue
}
& $SpLg -m pip freeze > ".pipeline-envs\sp_lightglue.freeze.txt"

if (-not $Cpu) {
    $cuda = & $XFeat -c "import torch; print(torch.cuda.is_available())"
    if ($cuda.Trim() -ne "True") {
        Write-Warning "torch reports no CUDA device in the matcher environment. The run will fall back to CPU and take considerably longer. Check the NVIDIA driver, or pass -Cpu to make that choice explicit."
    } else {
        Write-Host "  cuda available in the matcher environments"
    }
}

Write-Step "Recording this machine's matcher provenance"
if ($AcceptProvenance) {
    Invoke-Checked "recording provenance" {
        & $Eval -m warpaudit inspect-provenance --config $Config --write
    }
} else {
    & $Eval -m warpaudit inspect-provenance --config $Config
    Write-Host ""
    Write-Host "Rerun with -AcceptProvenance to record these pins and continue, or edit $Config by hand." -ForegroundColor Yellow
    exit 0
}

Write-Step "Probing configured pipeline blocks"
$PipelineIds = & $Eval -c "from warpaudit.config import load_config; import sys; print(*load_config(sys.argv[1]).common_block, sep='\n')" $Config
foreach ($pipeline in $PipelineIds) {
    Invoke-Checked "probing $pipeline" {
        & $Eval -m warpaudit probe-adapter --config $Config --pipeline $pipeline `
            --output "reports/probe_$pipeline.json"
    }
}
Write-Host "  all configured pipeline probes reproduce the pinned behaviour" -ForegroundColor Green

if ($SetupOnly) {
    Write-Step "Setup complete"
    Write-Host "Start the study with: .\scripts\run_study.ps1 -SignedOffBy `"Your Name`" -ReviewNote `"...`""
    exit 0
}

Write-Step "Running the study"
$studyArgs = @(
    "-m", "warpaudit", "run-study",
    "--config", $Config,
    "--download",
    "--acknowledge-fire-terms-unresolved"
)
if ($DryRun) { $studyArgs += "--dry-run" }
if ($SignedOffBy -and $ReviewNote) {
    $studyArgs += @("--signed-off-by", $SignedOffBy, "--review-note", $ReviewNote)
    if ($AcknowledgeInfeasible) { $studyArgs += "--acknowledge-infeasible" }
} elseif ($SignedOffBy -or $ReviewNote) {
    throw "-SignedOffBy and -ReviewNote must be given together; G1 is a reviewed gate."
} else {
    Write-Warning "No G1 sign-off given: the run will stop after the M2 feasibility screen. Rerun with -SignedOffBy and -ReviewNote to continue into the confirmatory evaluation."
}
if ($RefitBootstrap -gt 0) { $studyArgs += @("--refit-bootstrap", "$RefitBootstrap") }

if (-not (Test-Path "reports\study_runs")) {
    New-Item -ItemType Directory -Path "reports\study_runs" -Force | Out-Null
}
$transcript = "reports\study_runs\run-$(Get-Date -Format 'yyyyMMddTHHmmssZ').log"
Write-Host "  transcript: $transcript"
Write-Host ""

# The study writes progress to stdout and stage failures to stderr, and both
# belong in the transcript. Under Windows PowerShell 5.1 a native command's
# stderr arrives as an ErrorRecord, which the script-wide "Stop" preference
# would turn into a terminating error mid-run, so relax it for this call and
# read the real outcome from the exit code.
$previousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $Eval @studyArgs 2>&1 | Tee-Object -FilePath $transcript
    $status = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousPreference
}

Write-Host ""
if ($status -eq 0) {
    Write-Host "Study run completed. Summary: reports\STUDY_RUN.md" -ForegroundColor Green
} else {
    Write-Host "Study run finished with exit status $status. Summary: reports\STUDY_RUN.md" -ForegroundColor Red
    Write-Host "Stage logs are under reports\study_runs\. Rerunning this script resumes: every stage recomputes only what is missing."
}
exit $status
