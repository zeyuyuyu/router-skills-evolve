# Stage-2 SFT corpus (build at 2026-05-07T15:35:48Z)

Built from `data_arranged/` (369 verified-passing run dirs).

- Train rows: 6413
- Val rows: 394
- Eval task descriptors: 35
- Diag task descriptors: 5

Source data was read-only; SHA256s of every file consumed are recorded in
`audit/per_row_provenance.jsonl` and `_build_meta.json`.

## Limitations note

The collection runner used standard chat-completion API calls without
extended-thinking enabled. As a result, NO explicit reasoning / thinking
content is preserved in this corpus — only visible response text and tool
calls are available as SFT targets. Trained students will imitate behavior
but not chain-of-thought.

## Files

- `train.jsonl` / `val.jsonl` — SFT rows; format-neutral (no chat template applied).
- `eval_tasks.jsonl` / `diag_tasks.jsonl` — task descriptors for force-routed eval.
- `domain_assets/<domain>_(system.txt | tools.json | tool_examples.jsonl)` — single canonical assets per domain.
- `audit/` — disjointness, length stats, prompt-overlap diagnostic, per-row+per-run provenance.

## Provenance

Every row carries an inline `_p` block (~80 bytes) with row_id / split / domain /
task_id / seed / epoch / phase / step. Heavy fields (full source_path, file
SHA256s, msg indices, model_id) live in `audit/per_row_provenance.jsonl`,
joined by row_id. The reverse map `audit/per_run_step_emitted.json` answers
"which rows came from step k of run X?".

## Reproducibility

- git_sha: 3af58dcd757b5b5d8ab95796c4f75942b0455fea
- partition_sha256: 5d1983ba1461b0c5e819872a8eb7aaf6438e74a77b282b2fbc878ee5e09fe07a
- Build is deterministic (sorted iteration; rerun produces byte-identical output).
