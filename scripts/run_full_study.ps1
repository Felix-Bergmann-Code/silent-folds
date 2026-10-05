<# One-command Windows execution of the frozen factorized-stability study. #>

[CmdletBinding()]
param(
    [string]$Python = "py -3.11",
    [switch]$AcceptProvenance,
    [switch]$Cpu,
    [switch]$SkipSetup,
    [string]$SignedOffBy = "",
    [string]$ReviewNote = "",
    [string]$Config = "configs/full_study.yaml",
    [ValidateRange(1, 64)][int]$RegistrationWorkers = 4,
    [ValidateRange(1, 64)][int]$CpuWorkers = 16,
    [ValidateRange(1, 64)][int]$WorkerThreads = 1
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not $SkipSetup) {
    if (-not $AcceptProvenance) {
        throw "The first Windows run requires -AcceptProvenance to record the workstation's exact Python/CUDA pins."
    }
    $setup = @{ Config = $Config; Python = $Python; SetupOnly = $true }
    if ($AcceptProvenance) { $setup.AcceptProvenance = $true }
    if ($Cpu) { $setup.Cpu = $true }
    & "$PSScriptRoot\run_study.ps1" @setup
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$Eval = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Eval)) { throw "Evaluation environment absent; rerun without -SkipSetup." }
$RunConfig = (& $Eval -c "import json,sys; from pathlib import Path; from warpaudit.config import load_config; c=load_config(sys.argv[1]); print(json.dumps({'pipelines': c.common_block, 'paths': {k: str(v) for k,v in c.paths.resolve(Path.cwd()).items()}}))" $Config) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw "Unable to load the full-study configuration." }
$Pipelines = $RunConfig.pipelines
$ReportRoot = $RunConfig.paths.reports
$ManifestRoot = $RunConfig.paths.manifests

function Invoke-WarpAudit([string[]]$Arguments) {
    Write-Host "[$([DateTime]::UtcNow.ToString('o'))] warpaudit $($Arguments -join ' ')"
    $command = $Arguments[0]
    if ($command -in @("register", "e2")) {
        $Arguments += @("--workers", "$RegistrationWorkers", "--worker-threads", "$WorkerThreads")
    } elseif ($command -in @("features", "labels", "audit-data", "freeze-full-study", "evaluate-full-study")) {
        $Arguments += @("--workers", "$CpuWorkers", "--worker-threads", "$WorkerThreads")
        if ($command -eq "features") { $Arguments += @("--flush-every", "8") }
    }
    & $Eval -m warpaudit @Arguments
    if ($LASTEXITCODE -ne 0) { throw "warpaudit $($Arguments -join ' ') failed" }
}

# One orchestrator owns cache writes; worker processes only compute. Native
# library limits also propagate into each isolated matcher environment.
$env:OMP_NUM_THREADS = "$WorkerThreads"
$env:OPENBLAS_NUM_THREADS = "$WorkerThreads"
$env:MKL_NUM_THREADS = "$WorkerThreads"
$env:PYTHONUNBUFFERED = "1"
Write-Host "Execution: registration workers=$RegistrationWorkers; CPU workers=$CpuWorkers; native threads=$WorkerThreads"
Invoke-WarpAudit @("environment", "--config", $Config, "--output", (Join-Path $ReportRoot "environment.json"))
Invoke-WarpAudit @("audit-data", "--config", $Config)

foreach ($pipeline in $Pipelines) {
    Invoke-WarpAudit @("register", "--config", $Config, "--split", "development", "--pipeline", $pipeline, "--direction", "both")
}
foreach ($pipeline in $Pipelines) {
    Invoke-WarpAudit @("labels", "--config", $Config, "--split", "development", "--pipeline", $pipeline, "--direction", "canonical")
}
foreach ($pipeline in $Pipelines) {
    Invoke-WarpAudit @("e2", "--config", $Config, "--split", "development", "--pipeline", $pipeline)
}
Invoke-WarpAudit @("features", "--config", $Config, "--split", "development", "--families", "A", "B", "C", "D", "E1", "E2", "F", "G", "--direction", "canonical")
Invoke-WarpAudit @("plan-full-study", "--config", $Config)

if ($SignedOffBy -xor $ReviewNote) {
    throw "-SignedOffBy and -ReviewNote must be given together; G2 is a reviewed gate."
}
if (-not $SignedOffBy) {
    Write-Warning "Stopped at the prospective information gate. Review $ManifestRoot\full_study_information_plan.json, the dataset indexes, margins, and protocol; then rerun with -SkipSetup -SignedOffBy and -ReviewNote."
    exit 0
}

# This is the irreversible boundary: external outcomes remain unjoined until
# the complete configuration, development evidence, and code identity have
# been reviewed and frozen.
Invoke-WarpAudit @("freeze-full-study", "--config", $Config, "--signed-off-by", $SignedOffBy, "--review-note", $ReviewNote)

foreach ($pipeline in $Pipelines) {
    Invoke-WarpAudit @("register", "--config", $Config, "--split", "confirmatory", "--pipeline", $pipeline, "--direction", "both")
}
foreach ($pipeline in $Pipelines) {
    Invoke-WarpAudit @("e2", "--config", $Config, "--split", "confirmatory", "--pipeline", $pipeline)
}
Invoke-WarpAudit @("features", "--config", $Config, "--split", "confirmatory", "--families", "A", "B", "C", "D", "E1", "E2", "F", "G", "--direction", "canonical")
foreach ($pipeline in $Pipelines) {
    Invoke-WarpAudit @("labels", "--config", $Config, "--split", "confirmatory", "--pipeline", $pipeline, "--direction", "canonical")
}
Invoke-WarpAudit @("evaluate-full-study", "--config", $Config)
Write-Host "Full-study results: $ManifestRoot\full_study_results.json" -ForegroundColor Green
