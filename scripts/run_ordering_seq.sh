#!/bin/bash
# Sequential skill/LLM ordering comparison.
#   ORDERING=skill_first    Skill -> LLM -> Router
#   ORDERING=llm_first      LLM   -> Skill -> Router
# 4 cycles each. Strict serial. Single GPU (set via CUDA_VISIBLE_DEVICES).

set -euo pipefail

ORDERING="${ORDERING:-skill_first}"

cd /data0/home/zeyuwang/router-skills-evolve
export http_proxy=http://127.0.0.1:1080
export https_proxy=http://127.0.0.1:1080

DATA=/data0/home/zeyuwang/router-skills-evolve-data
WORK=/data0/home/zeyuwang/router-skills-evolve-runs/ordering/${ORDERING}
RES=/data0/home/zeyuwang/router-skills-evolve-results/ordering/${ORDERING}
mkdir -p "$WORK"/{skill_books,router_models,llm_adapters} "$RES"

SKILL_BOOK="$WORK/skill_books/skill_book.json"
rm -f "$SKILL_BOOK"

ROUTER_DATA=/data0/home/zeyuwang/router-skills-evolve-runs/joint_cycles/router_data
TRACES_FULL=/data0/home/zeyuwang/router-skills-evolve/data/traces/real_humaneval_30_90.jsonl
TASKS_FULL=/data0/home/zeyuwang/router-skills-evolve/data/HumanEval.jsonl
LLM_CHUNKS=/data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/chunks
BASE_LLM="Qwen/Qwen2.5-Coder-1.5B-Instruct"

LOG="$RES/ordering.log"
TIMINGS="$RES/cycle_timings.jsonl"
> "$LOG"; > "$TIMINGS"
echo "[$(date -u +%FT%TZ)] ORDERING=$ORDERING start, CUDA=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"

skill_update() {
  local k=$1
  python3 - <<PY >> "$LOG" 2>&1
import sys, json, os
sys.path.insert(0, ".")
from src import SkillBook
book = SkillBook()
if os.path.exists("$SKILL_BOOK"):
    book.load("$SKILL_BOOK")
new_start = 15 * ($k - 1)
new_end = 15 * $k
rows = []
with open("$TRACES_FULL") as f:
    for i, line in enumerate(f):
        if i < new_start: continue
        if i >= new_end: break
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
print(f"[skill cycle $k] applied={upd} sigs={len(book.skills)}")
PY
}

router_train_eval() {
  local k=$1
  local out="$WORK/router_models/cycle${k}"
  echo "[router cycle $k] start $(date -u +%FT%TZ)" >> "$LOG"
  python3 experiments/train_learnable_router.py \
    --traces "$ROUTER_DATA/cycle${k}_traces.jsonl" \
    --router-data "$ROUTER_DATA/cycle${k}_train.jsonl" \
    --tasks "$TASKS_FULL" \
    --base-model google/bert_uncased_L-2_H-128_A-2 \
    --output "$out" \
    --epochs 4 --batch-size 32 --eval-ratio 0.25 --class-weight balanced >> "$LOG" 2>&1
  python3 experiments/evaluate_learnable_router.py \
    --model "$out" \
    --traces "$ROUTER_DATA/cycle${k}_traces.jsonl" \
    --router-data "$ROUTER_DATA/bench_test.jsonl" \
    --tasks "$TASKS_FULL" \
    --output "$RES/cycle${k}_router_eval.json" >> "$LOG" 2>&1
  echo "[router cycle $k] done $(date -u +%FT%TZ)" >> "$LOG"
}

llm_train_eval() {
  local k=$1
  local out="$WORK/llm_adapters/cycle${k}"
  local prev=""
  if [[ $k -gt 1 ]]; then
    prev="--resume-from-adapter $WORK/llm_adapters/cycle$((k-1))"
  fi
  echo "[llm cycle $k] start $(date -u +%FT%TZ)" >> "$LOG"
  python3 experiments/train_small_model_grpo_local.py \
    --data "$LLM_CHUNKS/chunk_${k}.jsonl" \
    --base-model "$BASE_LLM" \
    --output "$out" \
    --limit 100 --epochs 1 --n-generations 4 \
    --max-new-tokens 192 --temperature 0.8 --top-p 0.95 \
    --lr 5e-6 --lora-r 16 --prompt-style qwen-chat \
    --reward-baseline group --reward-mode partial --kl-coef 0.05 \
    $prev >> "$LOG" 2>&1
  python3 experiments/evaluate_finetuned_model.py \
    --data /data0/home/zeyuwang/router-skills-evolve-data/mbpp_aug/test_eval_all.jsonl \
    --base-model "$BASE_LLM" --adapter "$out" \
    --output "$RES/cycle${k}_llm_eval.json" \
    --max-new-tokens 384 --prompt-style qwen-chat --limit 200 >> "$LOG" 2>&1
  echo "[llm cycle $k] done $(date -u +%FT%TZ)" >> "$LOG"
}

for k in 1 2 3 4; do
  echo "[$(date -u +%FT%TZ)] ===== cycle $k start =====" | tee -a "$LOG"
  T_START=$(date +%s.%N)

  case "$ORDERING" in
    skill_first)   skill_update $k; llm_train_eval $k; router_train_eval $k ;;
    llm_first)     llm_train_eval $k; skill_update $k; router_train_eval $k ;;
    *) echo "unknown ORDERING: $ORDERING" >&2; exit 1 ;;
  esac

  T_END=$(date +%s.%N)
  WALL=$(python3 -c "print(round($T_END - $T_START, 2))")
  echo "{\"cycle\": $k, \"ordering\": \"$ORDERING\", \"wall_sec\": $WALL}" >> "$TIMINGS"
  echo "[$(date -u +%FT%TZ)] ===== cycle $k done in ${WALL}s =====" | tee -a "$LOG"
done

echo "[$(date -u +%FT%TZ)] ORDERING=$ORDERING ALL 4 CYCLES DONE" | tee -a "$LOG"
