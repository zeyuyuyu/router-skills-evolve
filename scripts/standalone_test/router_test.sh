#!/usr/bin/env bash
# Standalone test: does the learned router help? Applies the trained router to a
# held-out trace set (with small/large outcomes) and compares always-small vs
# router-routed task-pass and cost. Offline (no GPU needed unless emb router).
# Usage: bash router_test.sh <router_dir> <heldout_traces.jsonl> [GPU]
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /shared_home/yuhang.yao/router-skills-evolve)"
RDIR="${1:?need router dir}"; TR="${2:?need heldout traces.jsonl}"; GPU="${3:-0}"
CUDA_VISIBLE_DEVICES=$GPU venv/bin/python - "$RDIR" "$TR" <<'PY'
import sys, json, joblib, numpy as np
sys.path.insert(0,".")
rdir, tr = sys.argv[1], sys.argv[2]
rows=[json.loads(l) for l in open(tr)]
so=np.array([bool(t.get("small_success")) for t in rows])
lo=np.array([bool(t.get("large_success")) for t in rows])
prompts=[t.get("prompt","") for t in rows]; n=len(rows)
pipe=joblib.load(f"{rdir}/router.joblib")
proba=pipe.predict_proba(prompts)[:,1]
thr=0.5
import os
tj=f"{rdir}/../router_threshold.json"
preds=(proba>=thr).astype(int)
used=preds==1
ok=np.where(used,lo,so)
cost=(used.sum()*0.01+(n-used.sum())*0.001)/(n*0.01)
print(f"[router_test] n={n}")
print(f"[router_test] always-small pass={so.mean():.4f}  always-large pass={lo.mean():.4f}")
print(f"[router_test] +router pass={ok.mean():.4f}  routed_large={used.mean():.4f}  cost_vs_large={cost:.4f}")
print(f"[router_test] router delta vs always-small: {(ok.mean()-so.mean()):+.4f}")
PY
