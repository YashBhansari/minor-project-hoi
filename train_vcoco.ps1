$ErrorActionPreference = 'Stop'

$work = $PSScriptRoot
$python = $env:UPT_PYTHON
if ([string]::IsNullOrWhiteSpace($python) -or -not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Set UPT_PYTHON to the full path of the compatible Python executable before training.'
}
$trainer = Join-Path $work 'train_upt_windows.py'
$initialization = Join-Path $work 'checkpoints\official\detr-r50-vcoco.pth'
$output = Join-Path $work 'outputs\training\vcoco'
$latest = Join-Path $output 'checkpoint_latest.pt'
$final = Join-Path $output 'checkpoint_final.pt'

foreach ($path in @($python, $trainer, $initialization)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required file not found: $path" }
}
New-Item -ItemType Directory -Force -Path $output | Out-Null

$mutex = [System.Threading.Mutex]::new($false, 'Local\UPT_VCOCO_Training')
$hasMutex = $false
try {
    try { $hasMutex = $mutex.WaitOne(0) }
    catch [System.Threading.AbandonedMutexException] { $hasMutex = $true }
    if (-not $hasMutex) { throw 'A V-COCO training process is already running.' }
    $arguments = @($trainer, '--dataset', 'vcoco', '--detr-checkpoint', $initialization,
        '--output-dir', $output, '--epochs', '20', '--batch-size', '1',
        '--checkpoint-interval', '250')
    if (Test-Path -LiteralPath $latest -PathType Leaf) {
        $arguments += @('--resume', $latest)
    } elseif (Test-Path -LiteralPath $final -PathType Leaf) {
        $arguments += @('--resume', $final)
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "V-COCO training stopped with exit code $LASTEXITCODE." }
} finally {
    if ($hasMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
