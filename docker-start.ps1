<#
  docker-start.ps1 - start the AudioPipeline containers WITHOUT ever rebuilding or downloading.

  Why: docker-compose.yml has a `build:` section, so `docker compose up -d` silently rebuilds
  (CUDA base image + PyTorch, many GB) whenever Docker can't find audiopipeline-backend:v1.
  This script only uses what is already on the machine, and tells you why if it can't.

  Usage:   .\docker-start.ps1              # start existing containers / images, never build
           .\docker-start.ps1 -Rebuild     # deliberately rebuild (the long download)
#>
param([switch]$Rebuild)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Fail($msg) { Write-Host "`n$msg" -ForegroundColor Red; exit 1 }

# 1. Docker engine reachable?
docker info --format '{{.ServerVersion}}' *> $null
if ($LASTEXITCODE -ne 0) { Fail "Docker engine is not reachable. Start Docker Desktop and wait until it says 'Engine running'." }

$ctx = (docker context show).Trim()
Write-Host "Docker context: $ctx"

if ($Rebuild) {
    Write-Host "Rebuilding on purpose (this downloads the CUDA image and PyTorch)..." -ForegroundColor Yellow
    docker compose up -d --build
    exit $LASTEXITCODE
}

# 2. Existing containers? Then just start them: no compose, no build, no pull.
$containers = @("audiopipeline-api", "audiopipeline-ui")
$existing = @($containers | Where-Object { (docker ps -a --filter "name=^$_$" --format '{{.Names}}') -eq $_ })
if ($existing.Count -eq $containers.Count) {
    Write-Host "Found existing containers: $($existing -join ', '). Starting them..." -ForegroundColor Green
    docker start @existing | Out-Null
    docker ps --filter "name=audiopipeline" --format "table {{.Names}}`t{{.Status}}`t{{.Ports}}"
    exit 0
}

# 3. No containers, but are the images here? Then create containers from them, never build.
$images = @("audiopipeline-backend:v1", "audiopipeline-frontend:v1")
$missing = @($images | Where-Object { -not (docker images -q $_) })
if ($missing.Count -eq 0) {
    Write-Host "Images found. Creating containers from them (no build, no download)..." -ForegroundColor Green
    docker compose up -d --no-build --pull never
    exit $LASTEXITCODE
}

# 4. Images are missing in this Docker. Explain instead of silently rebuilding.
Write-Host "`nThese images are not in the current Docker context ($ctx): $($missing -join ', ')" -ForegroundColor Yellow
Write-Host "That is why 'docker compose up -d' started building. Most common causes:"
Write-Host "  a) Docker Desktop switched image store: Settings > General > 'Use containerd for pulling and storing images'."
Write-Host "     If it was turned on (often by an update), turn it OFF, Apply & restart: your old images come back."
Write-Host "  b) A different Docker context is active. Contexts on this PC:"
docker context ls
Write-Host "     Switch with: docker context use <name>   (usually 'desktop-linux' or 'default')"
Write-Host "  c) The images were deleted (Docker Desktop 'Clean / Purge data', 'docker system prune -a', or a reset)."
Write-Host "     Then a rebuild is the only way back:  .\docker-start.ps1 -Rebuild"
Write-Host "`nImages with 'audiopipeline' in this context:"
docker images --format "{{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedSince}}" | Select-String audiopipeline
Fail "Nothing was built or downloaded."
