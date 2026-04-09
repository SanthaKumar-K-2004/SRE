$ErrorActionPreference = "Stop"

$SpaceBaseUrl = if ($env:SPACE_BASE_URL) { $env:SPACE_BASE_URL } else { "https://santhakumar-k-2004-sre-bench.hf.space" }
$SpaceRawInferenceUrl = if ($env:SPACE_RAW_INFERENCE_URL) { $env:SPACE_RAW_INFERENCE_URL } else { "https://huggingface.co/spaces/santhakumar-k-2004/sre-bench/raw/main/inference.py" }
$SpaceGitUrl = if ($env:SPACE_GIT_URL) { $env:SPACE_GIT_URL } else { "https://huggingface.co/spaces/santhakumar-k-2004/sre-bench" }

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message"
}

function Write-Pass {
    param([string]$Message)
    Write-Host "[PASS] $Message"
}

function Fail {
    param([string]$Message)
    throw "[FAIL] $Message"
}

Write-Info "Running local validation gates"
python -m pytest -q
python verify_sre_bench.py --gate phase1
python verify_sre_bench.py --gate phase2
python verify_sre_bench.py
Write-Pass "Local validation gates passed"

Write-Info "Checking GitHub main and Hugging Face Space main alignment"
git fetch origin main | Out-Null
$originSha = (git rev-parse origin/main).Trim()
$spaceSha = ((git ls-remote $SpaceGitUrl refs/heads/main) -split "\s+")[0]

if ([string]::IsNullOrWhiteSpace($spaceSha)) {
    Fail "Unable to resolve Hugging Face Space main"
}

if ($originSha -ne $spaceSha) {
    Fail "GitHub main ($originSha) and HF Space main ($spaceSha) are out of sync"
}

Write-Pass "GitHub main and HF Space main match at $originSha"

Write-Info "Checking live Space health, raw inference sync, and remote smoke"
python check_space_release.py --space-url $SpaceBaseUrl --raw-url $SpaceRawInferenceUrl
Write-Pass "Live Space release checks passed"

Write-Host ""
Write-Pass "Submission guard completed successfully"
Write-Info "Team lead can now click Update submission."
