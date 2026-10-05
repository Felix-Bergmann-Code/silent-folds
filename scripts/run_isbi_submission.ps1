<#
.SYNOPSIS
    One command for every remaining ISBI 2027 experiment, parallelized for a
    many-core CPU plus one CUDA GPU (tuned for i9-14900K + RTX 3090).

.DESCRIPTION
    From the repository root on the workstation holding the full-study caches:

        git pull
        .\scripts\run_isbi_submission.ps1

    Schedule:
      * GPU, in the background from the start: SuperRetina (official released
        model) in -GpuShards parallel processes sharing the GPU.
      * CPU, meanwhile, one process per logical core minus headroom:
          tests -> optimizer-warning audit -> Exp. A clearance (210 detector
          cells in parallel) -> Exp. B constrained RANSAC (all cached
          homographies, slowest cases first).
      * Then: SuperRetina evaluation (parallel), zip of all results.
    BLAS/OpenMP are pinned to one thread per worker process, so there is no
    oversubscription.

    Stage C needs the released SuperRetina weights, which are only on Google
    Drive: download SuperRetina.pth from
    https://drive.google.com/drive/folders/1h-MH3wEiN7BoLyMRjF1OAwABKqq6gVFL
    and save it as checkpoints\superretina\SuperRetina.pth. Without the file
    stage C is skipped with a message and the CPU stages still run. Every
    stage can be rerun; SuperRetina resumes where it stopped.

    Nothing here launches XFeat or LightGlue, and no file under warpaudit\ or
    scripts\matchers\ changes, so the cached registrations stay current.

.EXAMPLE
    .\scripts\run_isbi_submission.ps1
.EXAMPLE
    .\scripts\run_isbi_submission.ps1 -Only superretina -GpuShards 6
#>

[CmdletBinding()]
param(
    # CPU worker processes. Default: logical cores minus the GPU shards and two
    # for the OS (32 threads on an i9-14900K -> 26).
    [int]$Workers = 0,
    # SuperRetina processes sharing the GPU (each ~1-2 GB VRAM; the 3090 has 24 GB).
    [int]$GpuShards = 4,
    [ValidateSet("all", "tests", "warning", "clearance", "ransac", "superretina")]
    [string]$Only = "all",
    [string]$Python311 = "py -3.11",
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$Root = (Get-Location).Path
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Out = Join-Path $Root "reports\isbi_submission_latest"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Start-Transcript -Path (Join-Path $Out "run.log") -Append | Out-Null
$Results = [ordered]@{}
$RunAll = $Only -eq "all"
$Logical = [Environment]::ProcessorCount
if ($Workers -le 0) {
    $reserve = if ($RunAll -or $Only -eq "superretina") { $GpuShards } else { 0 }
    $Workers = [Math]::Max(1, $Logical - $reserve - 2)
}
Write-Host ("{0} logical CPUs -> {1} CPU workers, {2} GPU shards" -f $Logical, $Workers, $GpuShards)

# One BLAS/OpenMP thread per worker process (the Python scripts also set this).
foreach ($v in "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS") {
    Set-Item -Path "Env:$v" -Value "1"
}
$env:PYTHONUNBUFFERED = "1"

function Invoke-Stage([string]$Name, [scriptblock]$Body) {
    if (-not $RunAll -and $Only -ne $Name) { return }
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    $start = Get-Date
    try {
        $global:LASTEXITCODE = 0
        & $Body
        if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }
        $Results[$Name] = "ok ({0:N1} min)" -f ((Get-Date) - $start).TotalMinutes
    } catch {
        $Results[$Name] = "FAILED: $_"
        Write-Host "Stage $Name failed: $_" -ForegroundColor Red
    }
}

if (-not (Test-Path $Py)) { throw "Evaluation environment missing: $Py (see WORKSTATION.md)" }

# ---------------------------------------------------------------- GPU (background)
$SrOut = Join-Path $Out "superretina"
$SrProcs = @()
$SrStart = $null
if ($RunAll -or $Only -eq "superretina") {
    Write-Host "`n=== superretina: setup and launch ===" -ForegroundColor Cyan
    try {
        $SrRoot = Join-Path $Root "vendor\SuperRetina"
        $SrCommit = "338f041cc2ce86f39623e7da950b14f33bbc25df"
        $SrEnv = Join-Path $Root ".pipeline-envs\superretina"
        $SrPy = Join-Path $SrEnv "Scripts\python.exe"
        $Weights = Join-Path $Root "checkpoints\superretina\SuperRetina.pth"
        if (-not (Test-Path $Weights)) {
            New-Item -ItemType Directory -Force -Path (Split-Path $Weights) | Out-Null
            throw ("weights missing. Download SuperRetina.pth from " +
                   "https://drive.google.com/drive/folders/1h-MH3wEiN7BoLyMRjF1OAwABKqq6gVFL " +
                   "to $Weights, then: .\scripts\run_isbi_submission.ps1 -Only superretina")
        }
        if (-not (Test-Path $SrRoot)) {
            git clone https://github.com/ruc-aimc-lab/SuperRetina.git $SrRoot
        }
        git -C $SrRoot checkout --quiet $SrCommit
        if (-not (Test-Path $SrPy)) {
            Invoke-Expression "$Python311 -m venv `"$SrEnv`""
            & $SrPy -m pip install --upgrade pip
            if ($Cpu) {
                & $SrPy -m pip install torch==2.3.1 torchvision==0.18.1
            } else {
                & $SrPy -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
            }
            & $SrPy -m pip install numpy==1.26.4 opencv-python-headless==4.10.0.84 scipy matplotlib pillow tqdm pyyaml
            & $SrPy -m pip freeze | Set-Content (Join-Path $Out "superretina_env_freeze.txt")
        }
        if (-not $Cpu) {
            & $SrPy -c "import sys, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
            if ($LASTEXITCODE -ne 0) { throw "CUDA is not available in $SrEnv (use -Cpu to force CPU)" }
        }
        & $Py scripts\isbi_superretina_audit.py export-jobs --output (Join-Path $SrOut "jobs.json")
        if ($LASTEXITCODE -ne 0) { throw "export-jobs failed" }
        $device = if ($Cpu) { "cpu" } else { "cuda:0" }
        $cvThreads = [Math]::Max(1, [Math]::Floor(($Logical - $Workers) / $GpuShards))
        $SrStart = Get-Date
        for ($i = 0; $i -lt $GpuShards; $i++) {
            $srArgs = @(
                "scripts\superretina\run_superretina.py",
                "--jobs", (Join-Path $SrOut "jobs.json"),
                "--output", (Join-Path $SrOut ("homographies-{0:D2}.jsonl" -f $i)),
                "--superretina-root", $SrRoot, "--weights", $Weights, "--device", $device,
                "--shard", $i, "--num-shards", $GpuShards, "--cv-threads", $cvThreads
            )
            # SuperRetina's torch/OpenCV may use their own few threads.
            $env:OMP_NUM_THREADS = "$cvThreads"
            $quoted = ($srArgs | ForEach-Object { '"' + "$_" + '"' }) -join " "
            # Start-Process does not follow Set-Location: pass the repo root explicitly.
            $proc = Start-Process -FilePath $SrPy -ArgumentList $quoted -NoNewWindow -PassThru `
                -WorkingDirectory $Root `
                -RedirectStandardOutput (Join-Path $SrOut ("shard-{0:D2}.log" -f $i)) `
                -RedirectStandardError (Join-Path $SrOut ("shard-{0:D2}.err" -f $i))
            $null = $proc.Handle  # cache the handle so ExitCode is available later
            $SrProcs += $proc
            $env:OMP_NUM_THREADS = "1"
        }
        Write-Host "launched $GpuShards SuperRetina shards in the background (logs: $SrOut\shard-*.log)"
    } catch {
        $Results["superretina"] = "SKIPPED/FAILED: $_"
        Write-Host "SuperRetina: $_" -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------- CPU stages
Invoke-Stage "tests" {
    & $Py -m pytest -q tests\test_isbi_submission_experiments.py tests\test_pole_guard_ablation.py tests\test_projective_poles.py
}

Invoke-Stage "warning" {
    & $Py scripts\pole_guard_ablation.py --output reports\pole_guard_ablation_latest --warning-audit-only
}

Invoke-Stage "clearance" {
    & $Py scripts\isbi_submission_experiments.py clearance --workers $Workers
}

Invoke-Stage "ransac" {
    & $Py scripts\isbi_submission_experiments.py constrained-ransac --workers $Workers
}

# ---------------------------------------------------------------- GPU (join)
if ($SrProcs.Count -gt 0) {
    Write-Host "`n=== superretina: wait and evaluate ===" -ForegroundColor Cyan
    try {
        $SrProcs | Wait-Process
        $bad = @($SrProcs | Where-Object { $_.ExitCode -ne 0 })
        if ($bad.Count -gt 0) { throw "$($bad.Count) shard(s) failed; see $SrOut\shard-*.err" }
        # All CPU cores are free again for the evaluation re-fits.
        & $Py scripts\isbi_superretina_audit.py evaluate --workers ([Math]::Max(1, $Logical - 2))
        if ($LASTEXITCODE -ne 0) { throw "evaluate failed" }
        $Results["superretina"] = "ok ({0:N1} min)" -f ((Get-Date) - $SrStart).TotalMinutes
    } catch {
        $Results["superretina"] = "FAILED: $_"
        Write-Host "SuperRetina failed: $_" -ForegroundColor Red
    }
}

# ---------------------------------------------------------------- paper numbers
Write-Host "`n=== numbers ===" -ForegroundColor Cyan
& $Py scripts\build_isbi_numbers.py
if (Test-Path "paper\isbi_2027\numbers.tex") { Copy-Item "paper\isbi_2027\numbers.tex" $Out }

# ---------------------------------------------------------------- package
Write-Host "`n=== package ===" -ForegroundColor Cyan
$Zip = Join-Path $Root "reports\isbi_submission_results.zip"
$staging = Join-Path $env:TEMP "isbi_submission_pack"
Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $staging | Out-Null
Copy-Item -Recurse $Out (Join-Path $staging "isbi_submission_latest")
# Raw SuperRetina correspondences are large and reproducible; leave them out.
Get-ChildItem (Join-Path $staging "isbi_submission_latest\superretina") -Filter "homographies*.jsonl" -ErrorAction SilentlyContinue | Remove-Item -Force
$audit = Join-Path $Root "reports\pole_guard_ablation_latest\optimizer_warning_audit.json"
if (Test-Path $audit) { Copy-Item $audit $staging }
Compress-Archive -Force -Path (Join-Path $staging "*") -DestinationPath $Zip

Write-Host "`n=== summary ===" -ForegroundColor Cyan
$Results.GetEnumerator() | ForEach-Object { "{0,-12} {1}" -f $_.Key, $_.Value } | Tee-Object (Join-Path $Out "stage_summary.txt")
Write-Host "`nSend back: $Zip"
Write-Host "  or: git add reports\isbi_submission_latest paper\isbi_2027\numbers.tex reports\pole_guard_ablation_latest\optimizer_warning_audit.json; git commit -m 'ISBI submission experiments'; git push"
Stop-Transcript | Out-Null
