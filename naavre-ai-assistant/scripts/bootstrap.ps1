param(
    [string]$Python = 'python',
    [switch]$SkipFrontend
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$venvPython = Join-Path $projectRoot '.venv/Scripts/python.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Creating the virtual environment failed. Install Python 3.10+ first.' }
}
& $venvPython -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10+ required"'
if ($LASTEXITCODE -ne 0) { throw 'The copied virtual environment is unusable. Move it aside and rerun bootstrap to recreate it.' }

& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw 'Installing Python build tools failed.' }
& $venvPython -m pip install -e "${projectRoot}[test,build,notebook,analysis]"
if ($LASTEXITCODE -ne 0) { throw 'Installing the assistant and development dependencies failed.' }

if (-not $SkipFrontend) {
    $venvScripts = Split-Path -Parent $venvPython
    $previousPath = $env:Path
    Push-Location -LiteralPath $projectRoot
    try {
        $env:Path = "$venvScripts$([IO.Path]::PathSeparator)$previousPath"
        $npmCommand = Get-Command npm.cmd -ErrorAction Stop
        & $npmCommand.Source ci
        if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }
        & $npmCommand.Source run build:all
        if ($LASTEXITCODE -ne 0) { throw 'Building the frontend failed.' }
    }
    finally {
        $env:Path = $previousPath
        Pop-Location
    }
}
elseif (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'naavre_ai_assistant/labextension/package.json'))) {
    throw '-SkipFrontend requires an existing prebuilt frontend bundle. Run without this switch on a new source checkout.'
}
# setuptools editable installs omit data_files. A normal final installation is
# needed for Jupyter's shared frontend files and server auto-enable configuration.
& $venvPython -m pip install --no-deps --no-build-isolation --force-reinstall $projectRoot
if ($LASTEXITCODE -ne 0) { throw 'Registering the frontend and server extension failed.' }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Python dependency consistency check failed.' }
Write-Host 'Environment ready. Run naavre-ai-assistant/.venv/Scripts/python.exe naavre-ai-assistant/scripts/start_demo.py from this workspace.'
