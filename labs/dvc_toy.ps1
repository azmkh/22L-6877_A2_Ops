# Task 1.4 (toy) - Git + DVC on a tiny CSV, in a NEW empty folder outside this repo.
# Run from the activated .venv (dvc must be on PATH):   .\labs\dvc_toy.ps1
# Take a screenshot after each "==>" step.
param(
    [string]$ToyDir = "$HOME\dvc_toy_A2",
    [string]$RemoteDir = "$HOME\dvc_toy_A2_remote"
)
$ErrorActionPreference = "Stop"
function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

if (Test-Path $ToyDir) { throw "$ToyDir already exists - delete it or pass -ToyDir" }
New-Item -ItemType Directory $ToyDir | Out-Null
Set-Location $ToyDir

Step "git init + dvc init"
git init
dvc init
git commit -m "Initialise DVC"

Step "version 1 of data.csv, tracked by DVC"
"id,value`n1,10`n2,20" | Set-Content -Encoding ascii data.csv
dvc add data.csv
Get-Content data.csv.dvc          # the small pointer file that Git stores
git add data.csv.dvc .gitignore
git commit -m "Data v1: two rows"

Step "version 2: change the CSV"
Add-Content -Encoding ascii data.csv "3,30"
dvc add data.csv
git add data.csv.dvc
git commit -m "Data v2: three rows"
git log --oneline

Step "go back to version 1"
git checkout HEAD~1 -- data.csv.dvc
dvc checkout
Get-Content data.csv

Step "return to version 2"
git checkout HEAD -- data.csv.dvc
dvc checkout
Get-Content data.csv

Step "configure a remote, push, delete local data and cache, pull it back"
dvc remote add -d toyremote $RemoteDir
git add .dvc/config
git commit -m "Add DVC remote"
dvc push
Remove-Item data.csv
Remove-Item -Recurse -Force .dvc\cache
dvc pull
Get-Content data.csv

Write-Host "`nDone. To use DagsHub instead of a local folder, see RUNBOOK.md (Task 1.4)." -ForegroundColor Green
