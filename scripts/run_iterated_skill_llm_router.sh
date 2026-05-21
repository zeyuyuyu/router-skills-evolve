#!/bin/bash
# Multi-round Skill -> LLM -> Router iteration.
# Runs N cycles, each cycle does (in order):
#   1. SkillBook online update from a fresh trace batch
#   2. LLM GRPO continual training on chunk_(k mod 4 + 1), resuming previous adapter
#   3. Router BERT-tiny training on cumulative bench slice up to cycle k
# Persistent SkillBook + LLM adapter chain across all N cycles.
#
# Env:
#   NUM_CYCLES   default 8
#   RUN_TAG      subdir name (default iterated)

set -euo pipefail

NUM_CYCLES="${NUM_CYCLES:-8}"
RUN_TAG="${RUN_TAG:-iterated}"

cd /data0/home/zeyuwang/router-skills-evolve
export http_proxy=http://127.0.0.1:1080
export https_proxy=http://127.0.0.1:1080

DATA=/data0/home/zeyuwang/router-skills-evolve-data
WORK=/data0/home/zeyuwang/router-skills-evolve-runs/${RUN_TAG}
RES=/data0/home/zeyuwang/router-skills-evolve-results/${RUN_TAG}
mkdir -p "$WORK"/{skill_books,router_models,llm_adapters} "$RES"

SKILL_BOOK="$WORK/skill_books/skill_book.json"
rm -f "$SKILL_BOOK"

ROUTER_DATA=/data0/home/zeyuwang/router-skills-evolve-runs/joint_cycles/router_data
TRACES_FULL=/data0/home/zeyuwang/router-skills-evolve/data/traces/real_humaneval_30_90.jsonl
TASKS_FULL=/data0/home/zeyuwang/router-skills-evolve/data/HumanEval.jsonl
LLM_CHUNKS=/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/chunks
BASE_LLM="Qwen/Qwen2.5-Coder-1.5B-Instruct"

LOG="$RES/iterated.log"
TIMINGS="$RES/cycle_timings.jsonl"
> "$LOG"; > "$TIMINGS"
echo "[$(date -u +%FT%TZ)] Iterated Skill->LLM->Router for $NUM_CYCLES cycles, CUDA=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"

skill_update() {
  local k=$1
  # cycle through trace batches modulo (total/15)
  local total=$(wc -l < "$TRACES_FULL")
  local batches=$(( total / 15 ))
  local idx=$(( (k - 1) % batches ))
  local start=$(( idx * 15 ))
  local end=$(( start + 15 ))
  python3 - <<PY >> "$LOG" 2>&1
import sys, json, os
sys.path.insert(0, ".")
from src import SkillBook
book = SkillBook()
if os.path.exists("$SKILL_BOOK"):
    book.load("$SKILL_BOOK")
rows = []
with open("$TRACES_FULL") as f:
    for i, line in enumerate(f):
        if i < $start: continue
        if i >= $end: break
        rows.append(json.loads(line))
tasks = {}
with open("$TASKS_FULL") as f:
    for line in f:
        t = json.loads(line)
        tasks[t["task_id"]] = t
upd = 0
for r in rows:
    task = tasks.get(r["task_id"])
    if not task: continue
    model = r.get("final_model") or "openrouter/deepseek-v3.2"
    success = bool(r.get("final_success", False))
    book.update(task["prompt"], model, success, r["task_id"])
    upd += 1
book.save("$SKILL_BOOK")
print(f"[skill cycle $k] applied={upd} sigs={len(book.skills)} obs_total={sum(sum(v[1] for v in s.stats.values()) for s in book.skills.values())}")
PY
}

llm_train_eval() {
  local k=$1
  local chunk_idx=$(( ((k - 1) % 4) + 1 ))
  local out="$WORK/llm_adapters/cycle${k}"
  local prev=""
  if [[ $k -gt 1 ]]; then
    prev="--resume-from-adapter $WORK/llm_adapters/cycle$((k-1))"
  fi
  echo "[llm cycle $k chunk $chunk_idx] start $(date -u +%FT%TZ)" >> "$LOG"
  python3 experiments/train_small_model_grpo_local.py \
    --data "$LLM_CHUNKS/chunk_${chunk_idx}.jsonl" \
    --base-model "$BASE_LLM" \
    --output "$out" \
    --limit 100 --epochs 1 --n-generations 4 \
    --max-new-tokens 192 --temperature 0.8 --top-p 0.95 \
    --lr 5e-6 --lora-r 16 --prompt-style qwen-chat \
    --reward-baseline group --reward-mode partial --kl-coef 0.05 \
    $prev >> "$LOG" 2>&1
  python3 experiments/evaluate_finetuned_model.py \
    --data "$DATA/mbpp_aug/test_eval_all.jsonl" \
    --base-model "$BASE_LLM" --adapter "$out" \
    --output "$RES/cycle${k}_llm_eval.json" \
    --max-new-tokens 384 --prompt-style qwen-chat --limit 200 >> "$LOG" 2>&1
  echo "[llm cycle $k] done $(date -u +%FT%TZ)" >> "$LOG"
}

router_train_eval() {
  local k=$1
  # Router data cumulative: use cycle k's existing slice (cycles repeat modulo 4)
  local rd_idx=$(( ((k - 1) % 4) + 1 ))
  local out="$WORK/router_models/cycle${k}"
  echo "[router cycle $k slice $rd_idx] start $(date -u +%FT%TZ)" >> "$LOG"
  python3 experiments/train_learnable_router.py \
    --traces "$ROUTER_DATA/cycle${rd_idx}_traces.jsonl" \
    --router-data "$ROUTER_DATA/cycle${rd_idx}_train.jsonl" \
    --tasks "$TASKS_FULL" \
    --base-model google/bert_uncased_L-2_H-128_A-2 \
    --output "$out" \
    --epochs 4 --batch-size 32 --eval-ratio 0.25 --class-weight balanced \
    --seed $((42 + k)) >> "$LOG" 2>&1
  python3 experiments/evaluate_learnable_router.py \
    --model "$out" \
    --traces "$ROUTER_DATA/cycle${rd_idx}_traces.jsonl" \
    --router-data "$ROUTER_DATA/bench_test.jsonl" \
    --tasks "$TASKS_FULL" \
    --output "$RES/cycle${k}_router_eval.json" >> "$LOG" 2>&1
  echo "[router cycle $k] done $(date -u +%FT%TZ)" >> "$LOG"
}

for k in $(seq 1 $NUM_CYCLES); do
  echo "[$(date -u +%FT%TZ)] ===== cycle $k / $NUM_CYCLES start =====" | tee -a "$LOG"
  T_START=$(date +%s.%N)
  skill_update $k
  llm_train_eval $k
  router_train_eval $k
  T_END=$(date +%s.%N)
  WALL=$(python3 -c "print(round($T_END - $T_START, 2))")
  python3 - <<PY >> "$TIMINGS"
import json
print(json.dumps({"cycle": $k, "ordering": "iterated_skill_llm_router", "wall_sec": $WALL}))
PY
  echo "[$(date -u +%FT%TZ)] ===== cycle $k done in ${WALL}s =====" | tee -a "$LOG"
done

echo "[$(date -u +%FT%TZ)] ALL $NUM_CYCLES CYCLES DONE" | tee -a "$LOG"
