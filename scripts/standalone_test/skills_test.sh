#!/usr/bin/env bash
# Standalone test: does the distilled SkillBook procedure help the small model?
# Compares base-small vs base-small+procedure on the EVAL split, same harness.
# Usage: bash skills_test.sh <skillbook.json> [GPU] [HE_DATA]
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /shared_home/yuhang.yao/router-skills-evolve)"
SKILLBOOK="${1:?need skillbook.json}"; GPU="${2:-0}"; DATA="${3:-data/he_mbpp.jsonl}"
VENV=.vllm_cu12_venv; PORT=$((8110+GPU)); M=Qwen/Qwen2.5-Coder-1.5B-Instruct
CUDA_VISIBLE_DEVICES=$GPU PATH="$VENV/bin:$PATH" nohup "$VENV/bin/vllm" serve "$M" \
  --served-model-name "$M" --port $PORT --gpu-memory-utilization 0.85 --max-model-len 4096 \
  --dtype bfloat16 --trust-remote-code --enforce-eager > /tmp/skt_$PORT.log 2>&1 &
VP=$!; trap 'kill -9 $VP 2>/dev/null||true' EXIT
for _ in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; kill -0 $VP 2>/dev/null||{ tail /tmp/skt_$PORT.log; exit 1; }; sleep 3; done
HE_DATA="$PWD/$DATA" HE_MAX_REPAIR_TURNS=3 SKILLBOOK="$SKILLBOOK" \
HE_VLLM_MAP="{\"$M\": \"http://127.0.0.1:$PORT/v1\"}" \
venv/bin/python - <<'PY'
import os
from concurrent.futures import ThreadPoolExecutor
from src.pipeline.benches.humaneval.adapter import Adapter
from src.skills import SkillBook
M="Qwen/Qwen2.5-Coder-1.5B-Instruct"; ad=Adapter(); tasks=ad.load_tasks(10000, split="eval")
sb=SkillBook(); sb.load(os.environ["SKILLBOOK"])
def run(t,use): 
    ok,_,_=ad._gen_and_test(M,t,procedure=sb.get_procedure(t["prompt"]) if use else ""); return bool(ok)
def ev(use):
    with ThreadPoolExecutor(max_workers=64) as ex: r=list(ex.map(lambda t:run(t,use),tasks))
    return sum(r),len(r)
b,n=ev(False); s,_=ev(True)
print(f"[skills_test] base={b/n:.4f} ({b}/{n})  +skills={s/n:.4f} ({s}/{n})  delta={(s-b)/n:+.4f}")
PY
