param([Parameter(Mandatory)][ValidateSet('hicodet', 'vcoco')][string]$Dataset)
$ErrorActionPreference = 'Stop'

$python = if ($env:UPT_PYTHON) { $env:UPT_PYTHON } else { (Get-Command python -ErrorAction SilentlyContinue).Source }
$trainer = Join-Path $PSScriptRoot 'train_upt_windows.py'
$initialization = Join-Path $PSScriptRoot "checkpoints\detr-r50-$Dataset.pth"
$output = Join-Path $PSScriptRoot "outputs\training\$Dataset"
$latest = Join-Path $output 'checkpoint_latest.pt'
$final = Join-Path $output 'checkpoint_final.pt'
$workers = @{ hicodet = '4'; vcoco = '0' }[$Dataset]

if ([string]::IsNullOrWhiteSpace($python)) {
    throw 'Activate the virtual environment or set UPT_PYTHON to the Python executable.'
}
foreach ($path in @($python, $trainer, $initialization)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required file not found: $path" }
}
New-Item -ItemType Directory -Force -Path $output | Out-Null

$mutex = [System.Threading.Mutex]::new($false, "Local\UPT_${Dataset}_Training")
$hasMutex = $false
try {
    try { $hasMutex = $mutex.WaitOne(0) }
    catch [System.Threading.AbandonedMutexException] { $hasMutex = $true }
    if (-not $hasMutex) { throw "A $Dataset training process is already running." }
    $arguments = @($trainer, '--dataset', $Dataset, '--detr-checkpoint', $initialization,
        '--output-dir', $output, '--epochs', '20', '--batch-size', '16', '--micro-batch', '2',
        '--num-workers', $workers, '--checkpoint-interval', '250')
    foreach ($checkpoint in @($latest, $final)) {
        if (Test-Path -LiteralPath $checkpoint -PathType Leaf) {
            $arguments += @('--resume', $checkpoint)
            break
        }
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "$Dataset training stopped with exit code $LASTEXITCODE." }
} finally {
    if ($hasMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
