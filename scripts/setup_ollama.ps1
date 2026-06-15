param(
    [string]$GenerationModel = "qwen3:14b",
    [string]$EmbeddingModel = "mxbai-embed-large",
    [switch]$UseProjectModelStore
)

$ErrorActionPreference = "Stop"

if ($UseProjectModelStore) {
    $projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
    $modelDir = Join-Path $projectRoot ".ollama\models"
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
    Write-Host "Project-local model store requested: $modelDir"
    Write-Host "Start 'ollama serve' with OLLAMA_MODELS=$modelDir before pulling models."
    $env:OLLAMA_MODELS = $modelDir
}

ollama --version
ollama pull $GenerationModel
ollama pull $EmbeddingModel
ollama list
