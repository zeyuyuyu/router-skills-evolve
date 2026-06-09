#!/usr/bin/env bash
# tau2_train_wrapper.sh — wrap colleague's tau2_stage2 train_pipeline.sh for the
# scaling pipeline. Two modes:
#
#   MODE=scaling_traces     (default)  Train on this cycle's SFT pairs with a
#                                       small deterministic replay sample from
#                                       the colleague stage2_v1 corpus. This
#                                       keeps the MERA hard-example update fast
#                                       while preserving some anti-forgetting
#                                       coverage. Fails loudly if TRAINING_DATA
#                                       is empty — never silently uses the fixed
#                                       corpus. (Wired 2026-05-21; replay
#                                       bounded 2026-06-09.)
#
#   MODE=colleague_corpus              Train ONLY on the colleague's fixed
#                                       stage2_v1 corpus; ignores TRAINING_DATA.
#                                       Use to reproduce the pre-evolve baseline.
#
# Required env vars:
#   BUNDLE_ROOT          /path/to/experiments/tau2_stage2
#   RUN_CONFIG           one of runs/<NN>_<model>_<data>.yaml (sans extension)
#   TRAIN_OUTPUT_DIR     where to land train_outputs/<run_id>/
#
# Optional:
#   TRAINING_DATA        path to scaling traces' extract (mode=scaling_traces)
#   MODE                 scaling_traces | colleague_corpus  (default: scaling_traces)
#   SCALING_BASE_REPLAY_ROWS
#                        number of stage2_v1 rows to replay in scaling_traces.
#                        Default 512. Use all/full/-1 for old full-corpus mode;
#                        use 0 for hard examples only.
#   SCALING_BASE_REPLAY_SEED
#                        deterministic replay sample seed (default 1234)
#   SCALING_TRACE_REPEAT repeat each current-cycle hard example N times before
#                        handoff to train_all.sh (default 16)

set -euo pipefail

: "${BUNDLE_ROOT:?BUNDLE_ROOT must be set (path to experiments/tau2_stage2)}"
: "${RUN_CONFIG:?RUN_CONFIG must be set (e.g. 05_qwen3_5_4b_273)}"
: "${TRAIN_OUTPUT_DIR:?TRAIN_OUTPUT_DIR must be set}"
: "${MODE:=colleague_corpus}"

cd "$BUNDLE_ROOT"

case "$MODE" in
  colleague_corpus)
    echo "[tau2_train_wrapper] MODE=colleague_corpus  using stage2_v1 SFT corpus"
    # Pre-validate the run config exists
    if [[ ! -f "code/training/configs/runs/${RUN_CONFIG}.yaml" ]]; then
      echo "[tau2_train_wrapper] ERROR: code/training/configs/runs/${RUN_CONFIG}.yaml not found"
      echo "Available run configs:"
      ls -1 code/training/configs/runs/ | sed 's/^/  /'
      exit 2
    fi
    # Colleague's pipeline trains ALL 10 runs in plan_c_prime.yaml by default.
    # For the scaling pipeline we only want ONE. Filter via ONLY_RUN (train_all.sh honours ONLY_RUN, not PLAN_RUN_FILTER).
    ONLY_RUN="$RUN_CONFIG" \
      bash code/training/orchestration/train_pipeline.sh
    # Copy the result to TRAIN_OUTPUT_DIR so the rest of the scaling pipeline
    # has a stable path.
    src="$BUNDLE_ROOT/train_outputs/$RUN_CONFIG"
    mkdir -p "$TRAIN_OUTPUT_DIR"
    if [[ -d "$src/checkpoint-best" ]]; then
      ln -sfn "$src/checkpoint-best" "$TRAIN_OUTPUT_DIR/checkpoint-best"
      echo "[tau2_train_wrapper] linked $src/checkpoint-best -> $TRAIN_OUTPUT_DIR/checkpoint-best"
    else
      echo "[tau2_train_wrapper] WARN: $src/checkpoint-best missing; training may have failed"
      exit 3
    fi
    ;;

  scaling_traces)
    # Train on THIS cycle's extracted traces (closes the evolve loop on the LLM
    # track). Review 2026-05-21: the old stub silently fell back to
    # colleague_corpus, so per-cycle traces never reached SFT and the bug was
    # invisible. This mode now (a) requires non-empty TRAINING_DATA, (b) converts
    # it through the colleague's convert_to_prompt_completion.py, and (c) FAILS
    # LOUDLY if anything is missing — it never silently trains on the wrong data.
    : "${TRAINING_DATA:?MODE=scaling_traces requires TRAINING_DATA (set by phase3)}"
    if [[ ! -s "$TRAINING_DATA" ]]; then
      echo "[tau2_train_wrapper] FATAL: TRAINING_DATA=$TRAINING_DATA is missing or empty."
      echo "  This cycle produced no SFT pairs. Likely causes:"
      echo "   - no hard tasks (small-fail + large-OK) this cycle, or"
      echo "   - the bench adapter did not record 'large_completion' in traces."
      echo "  See docs/PIPELINE_AUDIT.md. Refusing to silently fall back to the"
      echo "  fixed colleague corpus (that is the bug we just fixed)."
      exit 4
    fi

    n_pairs=$(wc -l < "$TRAINING_DATA" | tr -d ' ')
    echo "[tau2_train_wrapper] MODE=scaling_traces  $n_pairs SFT pairs from $TRAINING_DATA"

    STAGE_DIR="$TRAIN_OUTPUT_DIR/scaling_sft"
    mkdir -p "$STAGE_DIR"
    STAGE2_ROWS="$STAGE_DIR/stage2_rows.jsonl"

    # 1. {prompt, completion} -> colleague stage-2 row format
    #    ({messages, _target_index, _p, domain}). _target_index points at the
    #    assistant turn so convert_to_prompt_completion masks the user prompt.
    "${PYTHON:-python3}" - "$TRAINING_DATA" "$STAGE2_ROWS" "${TAU2_DOMAIN:-retail}" <<'PY'
import json, sys
src, dst, domain = sys.argv[1], sys.argv[2], sys.argv[3]
n = 0
with open(src) as fh, open(dst, "w") as out:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        prompt = r.get("prompt") or r.get("instruction") or ""
        completion = r.get("completion") or r.get("output") or ""
        if not prompt or not completion:
            continue
        task_id = str(r.get("task_id", "") or f"row{n}")
        row_id = f"{domain}_{task_id}_scaling_traces"
        row = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "_target_index": 1,
            "domain": domain,
            "_p": {
                "row_id": row_id,
                "split": "TRAIN",
                "source": "scaling_traces",
                "domain": domain,
                "task_id": task_id,
                "task_uniqueness_key": f"{domain}:{task_id}",
                "seed": 0,
                "epoch_alias": "scaling",
                "phase": "scaling_traces",
                "locked_source": "scaling_traces",
                "step_1based": 1,
                "step_depth_frac": 1.0,
                "n_parallel_tool_calls": 0,
                "run_dir": f"{domain}/{task_id}_scaling_traces",
                "row_cost_usd": 0.0,
            },
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
print(f"[scaling_traces] wrote {n} stage-2 rows -> {dst}", file=sys.stderr)
PY

    # 2. Build the MERA hard-example update corpus.
    #
    # The earlier tau2 port concatenated the full 6,413-row stage2_v1 corpus
    # every cycle. That is safe but much heavier than the paper's "selected
    # hard examples" LLM-update track, especially for 35B/16k FSDP. Bound replay
    # by default and upweight the current-cycle rows so a small hard-example
    # pool is not drowned out by old data. Operators can restore the old behavior
    # with SCALING_BASE_REPLAY_ROWS=all.
    COLLEAGUE_TRAIN="$BUNDLE_ROOT/data_processed/stage2_v1/train.jsonl"
    COMBINED="$STAGE_DIR/train_augmented_stage2.jsonl"
    MIX_META="$STAGE_DIR/replay_mix_meta.json"
    BASE_REPLAY_ROWS="${SCALING_BASE_REPLAY_ROWS:-512}"
    BASE_REPLAY_SEED="${SCALING_BASE_REPLAY_SEED:-1234}"
    TRACE_REPEAT="${SCALING_TRACE_REPEAT:-16}"
    if [[ -s "$COLLEAGUE_TRAIN" ]]; then
      "${PYTHON:-python3}" - "$COLLEAGUE_TRAIN" "$STAGE2_ROWS" "$COMBINED" "$MIX_META" "$BASE_REPLAY_ROWS" "$BASE_REPLAY_SEED" "$TRACE_REPEAT" <<'PY'
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

base_path, stage_path, out_path, meta_path = map(Path, sys.argv[1:5])
replay_spec = sys.argv[5].strip().lower()
seed = int(sys.argv[6])
trace_repeat = max(1, int(sys.argv[7]))

def read_jsonl(path):
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def domain(row):
    meta = row.get("_p") or {}
    return str(row.get("domain") or meta.get("domain") or "unknown")

base_rows = read_jsonl(base_path)
trace_rows = read_jsonl(stage_path)

if replay_spec in {"all", "full", "-1"}:
    replay_rows = list(base_rows)
elif replay_spec in {"", "0", "none", "false"}:
    replay_rows = []
else:
    target = max(0, int(replay_spec))
    rng = random.Random(seed)
    by_domain = defaultdict(list)
    for row in base_rows:
        by_domain[domain(row)].append(row)
    for rows in by_domain.values():
        rng.shuffle(rows)

    replay_rows = []
    domains = sorted(by_domain)
    if target and domains:
        quota, rem = divmod(target, len(domains))
        leftovers = []
        for idx, dom in enumerate(domains):
            want = quota + (1 if idx < rem else 0)
            take = by_domain[dom][:want]
            replay_rows.extend(take)
            leftovers.extend(by_domain[dom][want:])
        if len(replay_rows) < target and leftovers:
            rng.shuffle(leftovers)
            replay_rows.extend(leftovers[: target - len(replay_rows)])
        rng.shuffle(replay_rows)

expanded_traces = []
for _ in range(trace_repeat):
    expanded_traces.extend(trace_rows)

with out_path.open("w") as out:
    for row in replay_rows:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in expanded_traces:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")

meta = {
    "base_source": str(base_path),
    "trace_source": str(stage_path),
    "base_total_rows": len(base_rows),
    "base_replay_spec": replay_spec,
    "base_replay_rows": len(replay_rows),
    "base_replay_seed": seed,
    "trace_rows": len(trace_rows),
    "trace_repeat": trace_repeat,
    "trace_expanded_rows": len(expanded_traces),
    "combined_rows": len(replay_rows) + len(expanded_traces),
    "base_replay_domains": dict(Counter(domain(r) for r in replay_rows)),
    "trace_domains": dict(Counter(domain(r) for r in trace_rows)),
}
meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(meta, ensure_ascii=False), file=sys.stderr)
PY
      n_base_total=$(wc -l < "$COLLEAGUE_TRAIN" | tr -d ' ')
      n_combined=$(wc -l < "$COMBINED" | tr -d ' ')
      n_replay=$("${PYTHON:-python3}" - "$MIX_META" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["base_replay_rows"])
PY
)
      n_trace_expanded=$("${PYTHON:-python3}" - "$MIX_META" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["trace_expanded_rows"])
PY
)
      echo "[tau2_train_wrapper] replay corpus: $n_replay/$n_base_total base replay + $n_trace_expanded scaling rows ($n_pairs x $TRACE_REPEAT) = $n_combined rows"
    else
      echo "[tau2_train_wrapper] WARN: colleague corpus $COLLEAGUE_TRAIN absent; training on scaling rows only ($n_pairs)"
      : > "$COMBINED"
      for _ in $(seq 1 "$TRACE_REPEAT"); do
        cat "$STAGE2_ROWS" >> "$COMBINED"
      done
      cat > "$MIX_META" <<EOF
{"base_source":"$COLLEAGUE_TRAIN","trace_source":"$STAGE2_ROWS","base_total_rows":0,"base_replay_rows":0,"trace_rows":$n_pairs,"trace_repeat":$TRACE_REPEAT,"trace_expanded_rows":$(wc -l < "$COMBINED" | tr -d ' '),"combined_rows":$(wc -l < "$COMBINED" | tr -d ' ')}
EOF
    fi

    # 3. Hand off to the colleague pipeline pointed at the augmented data via the
    #    SCALING_TRAIN_FILE_STAGE2 hook in train_all.sh. (train_all.sh fails loudly
    #    if the file is empty — no silent fallback to the fixed corpus.)
    SCALING_TRAIN_FILE_STAGE2="$COMBINED" \
      SCALING_OUTPUT_DIR="$TRAIN_OUTPUT_DIR" \
      ONLY_RUN="$RUN_CONFIG" \
      bash "$BUNDLE_ROOT/code/training/orchestration/train_pipeline.sh"
    src="$TRAIN_OUTPUT_DIR"
    if [[ -d "$src/checkpoint-best" ]]; then
      echo "[tau2_train_wrapper] scaling_traces trained at $src/checkpoint-best"
    else
      echo "[tau2_train_wrapper] FATAL: training produced no checkpoint-best at $src"
      exit 3
    fi
    ;;

  *)
    echo "[tau2_train_wrapper] Unknown MODE=$MODE (use colleague_corpus or scaling_traces)"
    exit 2
    ;;
esac
