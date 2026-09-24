# Windows 环境搭建脚本
Write-Host "=== JEV+MDP 环境搭建 ===" -ForegroundColor Green

# 创建虚拟环境
Write-Host "创建虚拟环境..." -ForegroundColor Cyan
python -m venv .venv

# 激活虚拟环境
Write-Host "激活虚拟环境..." -ForegroundColor Cyan
.\.venv\Scripts\Activate.ps1

# 安装依赖
Write-Host "安装依赖..." -ForegroundColor Cyan
pip install --upgrade pip
pip install torch>=2.4.0 transformers datasets peft trl pytest pyyaml rich

Write-Host "环境搭建完成！" -ForegroundColor Green
Write-Host "使用 .\.venv\Scripts\python <script> 运行脚本" -ForegroundColor Cyan
