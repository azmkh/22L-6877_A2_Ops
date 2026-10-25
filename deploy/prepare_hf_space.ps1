# Task 7 - assemble a Hugging Face Docker Space for the model API in build\hf_space.
#   1. python -m visionguard.registry export      (models\champion.keras + champion.json)
#   2. .\deploy\prepare_hf_space.ps1 -HfUser <your-hf-username>
#   3. follow the printed git commands
param(
    [Parameter(Mandatory = $true)][string]$HfUser,
    [string]$SpaceName = $(if ($env:PROJECT_NAME) { $env:PROJECT_NAME } else { "22L-6877_A2_Ops" })
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "build\hf_space"

if (-not (Test-Path "$root\models\champion.keras")) {
    throw "models\champion.keras not found - run: python -m visionguard.registry export"
}
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force "$out\src\visionguard", "$out\api", "$out\models" | Out-Null

Copy-Item "$root\api\Dockerfile" "$out\Dockerfile"
Copy-Item "$root\requirements-api.txt" $out
foreach ($f in "__init__.py", "preprocess.py", "serving.py", "validation.py", "web.py") {
    Copy-Item "$root\src\visionguard\$f" "$out\src\visionguard\"
}
Copy-Item "$root\api\app.py", "$root\api\__init__.py" "$out\api\"
Copy-Item "$root\models\champion.keras", "$root\models\champion.json" "$out\models\"

$readme = @"
---
title: $SpaceName
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# VisionGuard model API ($SpaceName)

- ``GET /health`` - model name, version and threshold
- ``POST /predict`` - multipart field ``image``, header ``X-API-Key``

The API key is stored as a Space **secret** named ``API_KEY`` (Settings -> Variables and secrets).
"@
# UTF-8 without BOM: Hugging Face must see "---" as the very first bytes of README.md.
[IO.File]::WriteAllText("$out\README.md", $readme, (New-Object System.Text.UTF8Encoding $false))

"*.keras filter=lfs diff=lfs merge=lfs -text" | Set-Content -Encoding ascii "$out\.gitattributes"

Write-Host "Space folder ready: $out" -ForegroundColor Green
Write-Host @"

Next steps:
  1. Create the Space: https://huggingface.co/new-space  -> name $SpaceName, SDK = Docker, hardware = CPU basic (free)
  2. In the Space: Settings -> Variables and secrets -> New secret  API_KEY = <long random value>
  3. Push the files (you need git-lfs: git lfs install):
       cd "$out"
       git init -b main
       git lfs install
       git remote add origin https://huggingface.co/spaces/$HfUser/$SpaceName
       git add . ; git commit -m "Deploy VisionGuard API"
       git push --force origin main      # password = a Hugging Face access token with write scope
  4. Watch the build logs in the Space page. API URL:  https://$($HfUser.ToLower())-$($SpaceName.ToLower().Replace('_','-')).hf.space
"@
