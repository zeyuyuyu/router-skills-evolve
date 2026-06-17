#!/usr/bin/env bash
# code/training/orchestration/tokenizer_reaudit.sh
# Re-audit token counts on actual training tokenizers (Spec 5.3.1).
#
# Schema reference (data_audit.py output):
#   - tokenizer_id: str
#   - n_rows_train: int
#   - per_domain_total_toks: { <domain>: { n, mean, p50, p75, p90, p95, p99, max } }
#   - per_domain_target_toks: same shape
#   - per_phase_total_toks: same shape
#   - per_stratum_total_toks / per_stratum_target_toks: same shape, keyed by domain/phase/depth_bucket
#   - per_run: { n_runs, n_steps_dist, sum_total_toks_dist, max_total_toks_dist, ... }
#   - totals: { naive_train_tokens_per_epoch, loss_target_tokens_per_epoch, sequence_packed_tokens_per_epoch, redundancy_factor_overall }
#   - sweep_band_simulation: per-band roll-ups
#   - inference_rollout_estimate: per-domain p95/p99/max train_total_toks
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:?set BUNDLE_ROOT to bundle path}"
# VENV defaults to the active conda env's bin (server case); falls back to the
# local-dev venv path; local-dev override via VENV=... still works.
VENV="${VENV:-${CONDA_PREFIX:+$CONDA_PREFIX/bin}}"
VENV="${VENV:-$BUNDLE_ROOT/code/.venv/bin}"
PY="$VENV/python"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: python interpreter not found at $PY" >&2
    echo "       Set VENV=/path/to/venv/bin to override." >&2
    exit 1
fi

cd "$BUNDLE_ROOT/code"

TOKENIZERS=(
    "Qwen/Qwen3.5-2B"
    "Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-9B"
    "Qwen/Qwen3.6-35B-A3B"
)

mkdir -p "$BUNDLE_ROOT/data_processed/stage2_v1/audit"

for TOK in "${TOKENIZERS[@]}"; do
    SLUG=$(echo "$TOK" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
    OUT="$BUNDLE_ROOT/data_processed/stage2_v1/audit/audit_for_training.${SLUG}.json"
    echo "=== Auditing $TOK -> $OUT ==="
    "$PY" -m scripts.training_prep.data_audit \
        --root "$BUNDLE_ROOT/data_processed/stage2_v1" \
        --tokenizer "$TOK" \
        --out "$OUT"
done

echo ""
echo "=== Summary: per-tokenizer max sequence length (per_domain_total_toks[*].max) ==="
BUNDLE_ROOT="$BUNDLE_ROOT" "$PY" - <<'EOF'
import json, glob, os
root = os.environ["BUNDLE_ROOT"]
pattern = f"{root}/data_processed/stage2_v1/audit/audit_for_training.qwen*.json"
for p in sorted(glob.glob(pattern)):
    name = os.path.basename(p)
    with open(p) as f:
        d = json.load(f)
    pdt = d.get("per_domain_total_toks") or {}
    if pdt:
        per_dom = {dom: stats.get("max") for dom, stats in pdt.items()}
        global_max = max((v for v in per_dom.values() if v is not None), default=None)
        print(f"  {name}:")
        print(f"    tokenizer_id      = {d.get('tokenizer_id')}")
        print(f"    n_rows_train      = {d.get('n_rows_train')}")
        print(f"    global max total  = {global_max}")
        for dom, mx in sorted(per_dom.items()):
            print(f"    domain {dom:<8s} max = {mx}")
    else:
        print(f"  {name}: (no per_domain_total_toks; top-level keys = {sorted(d.keys())})")
EOF
