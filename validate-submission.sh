#!/usr/bin/env bash
# Final release guard for SRE-Bench submissions.

set -euo pipefail

SPACE_BASE_URL="${SPACE_BASE_URL:-https://santhakumar-k-2004-sre-bench.hf.space}"
SPACE_RAW_INFERENCE_URL="${SPACE_RAW_INFERENCE_URL:-https://huggingface.co/spaces/santhakumar-k-2004/sre-bench/raw/main/inference.py}"
SPACE_GIT_URL="${SPACE_GIT_URL:-https://huggingface.co/spaces/santhakumar-k-2004/sre-bench}"

info() {
  printf '[INFO] %s\n' "$1"
}

pass() {
  printf '[PASS] %s\n' "$1"
}

fail() {
  printf '[FAIL] %s\n' "$1" >&2
  exit 1
}

if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  fail "python or python3 is required"
fi

export PYTHONUTF8=1
export PYTHONPATH=.

info "Running local validation gates"
"$PYTHON" -m pytest -q
"$PYTHON" verify_sre_bench.py --gate phase1
"$PYTHON" verify_sre_bench.py --gate phase2
"$PYTHON" verify_sre_bench.py
pass "Local validation gates passed"

command -v git >/dev/null 2>&1 || fail "git is required for ref-alignment checks"

info "Checking GitHub main and Hugging Face Space main alignment"
git fetch origin main >/dev/null
ORIGIN_SHA="$(git rev-parse origin/main)"
SPACE_SHA="$(git ls-remote "$SPACE_GIT_URL" refs/heads/main | awk '{print $1}')"

[ -n "$SPACE_SHA" ] || fail "Unable to resolve Hugging Face Space main"

if [ "$ORIGIN_SHA" != "$SPACE_SHA" ]; then
  fail "GitHub main ($ORIGIN_SHA) and HF Space main ($SPACE_SHA) are out of sync"
fi
pass "GitHub main and HF Space main match at $ORIGIN_SHA"

info "Checking live Space health, raw inference sync, and remote smoke"
"$PYTHON" check_space_release.py \
  --space-url "$SPACE_BASE_URL" \
  --raw-url "$SPACE_RAW_INFERENCE_URL"
pass "Live Space release checks passed"

printf '\n'
pass "Submission guard completed successfully"
info "Team lead can now click Update submission."
