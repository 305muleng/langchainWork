$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Missing .venv. Follow README setup instructions first.'
}
foreach ($taskDirectory in @('.tmp', '.cache\pip', '.cache\tiktoken', 'logs')) {
    New-Item -ItemType Directory -Path (Join-Path $taskRoot $taskDirectory) -Force | Out-Null
}
$env:TEMP = Join-Path $taskRoot '.tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $taskRoot '.cache\pip'
$env:TIKTOKEN_CACHE_DIR = Join-Path $taskRoot '.cache\tiktoken'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
$env:LANGSMITH_TRACING = 'false'
Set-Location -LiteralPath $taskRoot
& $taskPython -B -m streamlit run app.py
