#!/usr/bin/env bash
# Standalone test: does LLM training (SFT adapter) help the small model?
# Compares base-small vs SFT-small on the EVAL split (no skills, same harness).
# Usage: bash llm_test.sh <sft_adapter_dir> [GPU] [HE_DATA]
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /shared_home/yuhang.yao/router-skills-evolve)"
SFT="${1:?need SFT adapter dir}"; GPU="${2:-0}"; DATA="${3:-data/he_mbpp.jsonl}"
VENV=.vllm_cu12_venv; PORT=$((8120+GPU)); M=Qwen/Qwen2.5-Coder-1.5B-Instruct
CUDA_VISIBLE_DEVICES=$GPU PATH="$VENV/bin:$PATH" nohup "$VENV/bin/vllm" serve "$M" \
  --served-model-name "$M" --port $PORT --gpu-memory-utilization 0.85 --max-model-len 4096 \
  --dtype bfloat16 --trust-remote-code --enforce-eager --enable-lora --max-lora-rank 16 \
  --lora-modules "sft=$SFT" > /tmp/llt_$PORT.log 2>&1 &
VP=$!; trap 'kill -9 $VP 2>/dev/null||true' EXIT
for _ in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; kill -0 $VP 2>/dev/null||{ tail /tmp/llt_$PORT.log; exit 1; }; sleep 3; done
HE_DATA="$PWD/$DATA" HE_MAX_REPAIR_TURNS=3 \
HE_VLLM_MAP="{\"$M\": \"http://127.0.0.1:$PORT/v1\", \"sft\": \"http://127.0.0.1:$PORT/v1\"}" \
venv/bin/python - <<'PY'
from concurrent.futures import ThreadPoolExecutor
from src.pipeline.benches.humaneval.adapter import Adapter
ad=Adapter(); tasks=ad.load_tasks(10000, split="eval")
def run(t,m): ok,_,_=ad._gen_and_test(m,t,procedure=""); return bool(ok)
def ev(m):
    with ThreadPoolExecutor(max_workers=64) as ex: r=list(ex.map(lambda t:run(t,m),tasks))
    return sum(r),len(r)
b,n=ev("Qwen/Qwen2.5-Coder-1.5B-Instruct"); s,_=ev("sft")
print(f"[llm_test] base={b/n:.4f} ({b}/{n})  +SFT={s/n:.4f} ({s}/{n})  delta={(s-b)/n:+.4f}")
PY
