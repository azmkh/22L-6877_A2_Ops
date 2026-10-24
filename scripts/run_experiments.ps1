# Task 3.5 - the experiment grid. Each run changes ONE thing relative to exp2 (the reference).
# Run from the repo root with the .venv active:   .\scripts\run_experiments.ps1
# Then:  python -m visionguard.compare_runs --log
$ErrorActionPreference = "Stop"

$runs = @(
    @("exp1_baseline_cnn",     "--model baseline --lr 1e-3 --epochs 40 --patience 10"),  # architecture: own CNN (from scratch needs longer)
    @("exp2_mnv2_reference",   "--model transfer --backbone mobilenetv2 --lr 1e-3"),  # reference
    @("exp3_mnv2_lr1e-4",      "--model transfer --backbone mobilenetv2 --lr 1e-4"),  # learning rate
    @("exp4_mnv2_noaug",       "--model transfer --backbone mobilenetv2 --lr 1e-3 --no-augment"),  # augmentation
    @("exp5_mnv2_frozen_only", "--model transfer --backbone mobilenetv2 --lr 1e-3 --unfreeze-layers 0"),  # frozen layers
    @("exp6_mnv2_160px",       "--model transfer --backbone mobilenetv2 --lr 1e-3 --img-size 160"),  # image size
    @("exp7_effnetb0",         "--model transfer --backbone efficientnetb0 --lr 1e-3"),  # backbone
    @("exp8_mnv2_autolog",     "--model transfer --backbone mobilenetv2 --lr 1e-3 --tracking autolog")  # Task 3.3 autolog
)

foreach ($r in $runs) {
    $name, $argsLine = $r
    Write-Host "`n===== $name : $argsLine =====" -ForegroundColor Cyan
    $argList = $argsLine -split " "
    python -m visionguard.train @argList --run-name $name --note "Task 3.5 grid: $argsLine"
    if ($LASTEXITCODE -ne 0) { throw "$name failed" }
}
Write-Host "`nAll runs finished. Next: python -m visionguard.compare_runs --log" -ForegroundColor Green
