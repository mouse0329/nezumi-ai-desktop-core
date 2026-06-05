# Build litert_lm_main for nezumi-ai-desktop-core (Windows)
$ErrorActionPreference = "Stop"

$GitBash = "$env:LOCALAPPDATA\Programs\Git\bin\bash.exe"
if (-not (Test-Path $GitBash)) {
    $GitBash = "C:\Program Files\Git\bin\bash.exe"
}
if (-not (Test-Path $GitBash)) {
    throw "Git Bash not found. Install Git for Windows and set BAZEL_SH."
}

$env:BAZEL_SH = $GitBash
$Root = Join-Path $PSScriptRoot ".." "LiteRT-LM" | Resolve-Path

Push-Location $Root
try {
    bazelisk --output_user_root=C:/bzl-user --output_base=C:/bzl `
        build //runtime/engine:litert_lm_main --config=windows `
        --shell_executable="$GitBash" `
        --copt=/utf-8 --host_copt=/utf-8
    $Exe = "C:/bzl/execroot/litert_lm/bazel-out/x64_windows-opt/bin/runtime/engine/litert_lm_main.exe"
    if (Test-Path $Exe) {
        Write-Host "Built: $Exe"
        Write-Host "Set for cargo: `$env:LITERT_LM_MAIN = '$Exe'"
    }
} finally {
    Pop-Location
}
