#!/usr/bin/env python3
"""Generate a markdown report of recent experiments."""
import datetime
import json
from pathlib import Path

STATE_PATH = Path("/data0/home/zeyuwang/auto_research/state.json")
REPORTS_DIR = Path("/data0/home/zeyuwang/auto_research/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    state = json.loads(STATE_PATH.read_text())
    today = datetime.datetime.utcnow().date().isoformat()
    out = REPORTS_DIR / f"daily-{today}.md"

    history = state.get("history", [])
    today_runs = [h for h in history if h.get("finished_at", "")[:10] == today]
    running = state.get("running", [])
    queue = state.get("queue", [])

    lines: list[str] = []
    lines.append(f"# Auto-research daily report  {today}")
    lines.append("")
    lines.append(f"- Running: **{len(running)}**")
    lines.append(f"- Finished today: **{len(today_runs)}**")
    lines.append(f"- In queue: **{len(queue)}**")
    lines.append(f"- Lifetime history: **{len(history)}**")
    lines.append("")

    if today_runs:
        lines.append("## Finished today")
        lines.append("")
        lines.append("| ID | Kind | Status | Headline metric |")
        lines.append("| --- | --- | --- | --- |")
        for h in today_runs:
            res = h.get("result", {}) or {}
            kind = res.get("kind", "?")
            status = res.get("status", "?")
            headline = res.get("final_pass_rate") or res.get("results") or "-"
            if isinstance(headline, list):
                headline = f"{len(headline)} sub-results"
            lines.append(f"| {h['id']} | {kind} | {status} | {headline} |")
        lines.append("")

    if running:
        lines.append("## Currently running")
        lines.append("")
        for r in running:
            lines.append(f"- `{r['id']}` (pid {r.get('pid')}, GPU {r.get('gpus')}, started {r.get('started_at')})")
        lines.append("")

    if queue:
        lines.append("## Queue (top 10 by priority)")
        lines.append("")
        top = sorted(queue, key=lambda e: -e.get("priority", 0))[:10]
        for e in top:
            lines.append(f"- p={e.get('priority',0)} `{e['id']}` — {e.get('rationale','(no rationale)')}")
        lines.append("")

    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
