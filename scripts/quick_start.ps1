$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (!(Test-Path .venv)) {
    Write-Host '[quick_start] creating venv'
    c:/Users/22130/Desktop/worksapce/.venv/Scripts/python.exe -m venv .venv
}

Write-Host '[quick_start] activating venv'
& .\.venv\Scripts\Activate.ps1

Write-Host '[quick_start] installing dependencies'
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host '[quick_start] downloading market data'
python scripts/download_market_data.py --config configs/default.yaml

Write-Host '[quick_start] running train/eval pipeline'
python src/rl_agent_research/orchestrator.py --config configs/default.yaml

Write-Host '[quick_start] tip: for DeepSeek-assisted planning, run:'
Write-Host '  python src/rl_agent_research/orchestrator_llm.py --config configs/default.yaml'

Write-Host '[quick_start] done'
