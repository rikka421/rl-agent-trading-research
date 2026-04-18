$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (!(Test-Path .venv)) {
    Write-Error '.venv not found. Please run scripts/quick_start.ps1 first.'
}

& .\.venv\Scripts\Activate.ps1

if (-not $env:DEEPSEEK_API_KEY) {
    Write-Host '[run_llm_mode] DEEPSEEK_API_KEY is not set. LLM planning will be skipped.'
}

python src/rl_agent_research/orchestrator_llm.py --config configs/default.yaml
