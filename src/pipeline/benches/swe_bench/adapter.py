"""SWE-Bench adapter — STUB.

Status: NOT IMPLEMENTED. Use `--bench tau2_bench` for now.

To implement SWE-Bench support, fill in `load_tasks` and `run_task_pair`
following the BenchAdapter protocol in `src/pipeline/benches/__init__.py`.

────────────────────────────── Implementation TODO ──────────────────────────────
Estimated effort: 2-3 engineer-days.

1. Pick a SWE-Bench variant:
     • SWE-Bench Lite  (~300 tasks, faster, recommended for first pass)
     • SWE-Bench Verified (~500 tasks, higher quality)
     Pin a HuggingFace snapshot SHA so results are reproducible.

2. Docker harness (REQUIRED — SWE-Bench tasks need isolated repo sandboxes):
     • Reuse the official SWE-Bench evaluation/ Docker images
     • One container per task instance with the pre-patched repo
     • Mount the agent's model output, run `git apply` + `pytest`
     • Tasks need ~5-15 min each; set `SWE_BENCH_TASK_TIMEOUT=1800`

3. `load_tasks(n, split)`:
     • Download SWE-Bench-Lite from HF: `princeton-nlp/SWE-bench_Lite`
     • Each task dict must contain: instance_id, repo, base_commit,
       problem_statement, test_patch (for eval), and the FAIL_TO_PASS list.
     • Map to scaling-pipeline schema:
         {"task_id": instance_id, "prompt": problem_statement, "repo": repo, ...}

4. `run_task_pair(task, small_model, large_model, cycle)`:
     • Spin up Docker container with the task's base_commit checked out
     • Drive the agent (small model first; on fail, fall back to large)
     • For each model: emit a generated patch
     • Apply patch + run FAIL_TO_PASS tests → success/fail signal
     • Compute cost from prompt+completion tokens × per-model rate
     • Return trace row matching the schema in __init__.py.

5. Cost-control note:
     SWE-Bench per-task cost is ~1-2 orders of magnitude higher than tau2-bench
     (long repo contexts, many tool calls). Recommend running on a subset of
     ~50-100 tasks for first scaling experiments, expand once router converges.

6. Reference implementations:
     • https://github.com/princeton-nlp/SWE-bench (official harness)
     • https://github.com/SWE-agent/SWE-agent (agent framework you can wrap)

────────────────────────────────────────────────────────────────────────────────
"""


class Adapter:
    def load_tasks(self, n, split="train"):
        raise NotImplementedError(
            "SWE-Bench adapter is a stub. See module docstring for implementation TODO. "
            "For now use --bench tau2_bench."
        )

    def run_task_pair(self, task, small_model, large_model, cycle):
        raise NotImplementedError(
            "SWE-Bench adapter is a stub. See module docstring for implementation TODO."
        )
