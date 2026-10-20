param(
    [ValidateSet('qwen3-coder:30b', 'qwen3.8:27b')]
    [string]$Model = 'qwen3.8:27b',
    [ValidateSet('auto', 'true', 'false')]
    [string]$Think = 'false',
    [ValidatePattern('^[a-z][a-z0-9_-]{0,63}(,[a-z][a-z0-9_-]{0,63})*$')]
    [string]$Labs = 'plants'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$venvPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw 'Project environment missing. Run naavre-ai-assistant/scripts/bootstrap.ps1 first.'
}
& $venvPython (Join-Path $PSScriptRoot 'start_demo.py') --model $Model --think $Think --labs $Labs
exit $LASTEXITCODE
