# JEV+MDP Web 服务启动脚本
# 用法: .\scripts\serve.ps1

$env:HF_HUB_OFFLINE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"

# 切换到项目根目录（$PSScriptRoot 即 scripts 目录，其上级为项目根；
# 若脚本路径解析失败则回退到当前工作目录）
if ($PSScriptRoot -and (Split-Path -Parent $PSScriptRoot)) {
    $projectRoot = Split-Path -Parent $PSScriptRoot
} else {
    $projectRoot = (Get-Location).Path
}
Set-Location $projectRoot

# 检查虚拟环境
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    Write-Host "使用虚拟环境: .venv" -ForegroundColor Cyan
    & $venvPython -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
} else {
    Write-Host "使用系统 Python" -ForegroundColor Yellow
    python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
}
