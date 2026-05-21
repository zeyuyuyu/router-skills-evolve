#!/usr/bin/env python3
"""arxiv hotspot scraper for LLM research.

Pulls recent papers from cs.CL, cs.LG, cs.AI matching LLM keywords.
Saves to /data0/home/zeyuwang/auto_research/hotspots/YYYY-MM-DD.json

Free arxiv API, no auth. Rate limit: 3 sec between queries.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HOTSPOTS_DIR = Path("/data0/home/zeyuwang/auto_research/hotspots")
HOTSPOTS_DIR.mkdir(parents=True, exist_ok=True)

# Keyword groups; a paper qualifies if title/abstract hits ANY of these tags.
KEYWORD_GROUPS = {
    "rl_align": ["RLHF", "DPO", "GRPO", "PPO", "reward model", "RLAIF", "preference learning"],
    "continual": ["continual learning", "catastrophic forgetting", "lifelong learning", "online learning"],
    "moe_routing": ["mixture of experts", "MoE", "expert routing", "model routing", "router"],
    "agent": ["agent", "tool use", "tool calling", "ReAct", "scaffolding"],
    "code_gen": ["code generation", "program synthesis", "MBPP", "HumanEval", "LiveCodeBench"],
    "rag_long_ctx": ["retrieval augmented", "RAG", "long context", "long-context"],
    "training_eff": ["LoRA", "adapter", "PEFT", "parameter efficient", "QLoRA"],
    "eval": ["evaluation benchmark", "leaderboard", "contamination", "data contamination"],
    "scaling": ["scaling law", "data scaling", "compute optimal", "chinchilla"],
}

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_arxiv(category: str, max_results: int = 100) -> list[dict]:
    query = f"cat:{category}"
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "auto-research/0.1 (zeyu)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
    root = ET.fromstring(body)
    entries = []
    for entry in root.findall("atom:entry", NS):
        eid = entry.findtext("atom:id", default="", namespaces=NS).strip()
        title = (entry.findtext("atom:title", default="", namespaces=NS) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=NS) or "").strip()
        published = entry.findtext("atom:published", default="", namespaces=NS).strip()
        authors = [
            a.findtext("atom:name", default="", namespaces=NS).strip()
            for a in entry.findall("atom:author", NS)
        ]
        entries.append({
            "id": eid,
            "title": re.sub(r"\s+", " ", title),
            "summary": re.sub(r"\s+", " ", summary),
            "published": published,
            "authors": authors[:5],
            "category": category,
        })
    return entries


def tag_paper(title: str, summary: str) -> list[str]:
    text = (title + " " + summary).lower()
    tags = []
    for group, kws in KEYWORD_GROUPS.items():
        for kw in kws:
            if kw.lower() in text:
                tags.append(group)
                break
    return tags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=["cs.CL", "cs.LG", "cs.AI"])
    parser.add_argument("--max-per-cat", type=int, default=100)
    parser.add_argument("--days", type=int, default=2, help="only keep papers from last N days")
    args = parser.parse_args()

    all_papers: list[dict] = []
    seen_ids = set()
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=args.days)

    for cat in args.categories:
        print(f"[arxiv] fetching {cat}...")
        try:
            papers = fetch_arxiv(cat, args.max_per_cat)
        except Exception as e:
            print(f"  failed: {e}")
            time.sleep(3)
            continue
        for p in papers:
            if p["id"] in seen_ids:
                continue
            seen_ids.add(p["id"])
            try:
                pub = datetime.datetime.fromisoformat(p["published"].replace("Z", "+00:00"))
                if pub.replace(tzinfo=None) < cutoff:
                    continue
            except Exception:
                pass
            p["tags"] = tag_paper(p["title"], p["summary"])
            if not p["tags"]:
                continue  # not LLM-relevant
            all_papers.append(p)
        time.sleep(3)  # arxiv asks for 3-sec delay

    today = datetime.datetime.utcnow().date().isoformat()
    out_path = HOTSPOTS_DIR / f"{today}.json"
    out_path.write_text(json.dumps({
        "date": today,
        "n_papers": len(all_papers),
        "n_categories": len(args.categories),
        "categories": args.categories,
        "papers": all_papers,
    }, indent=2, ensure_ascii=False))
    print(f"wrote {len(all_papers)} LLM-relevant papers to {out_path}")

    by_tag: dict[str, int] = {}
    for p in all_papers:
        for t in p["tags"]:
            by_tag[t] = by_tag.get(t, 0) + 1
    print()
    print("by tag:")
    for t, n in sorted(by_tag.items(), key=lambda x: -x[1]):
        print(f"  {t:20s} {n}")


if __name__ == "__main__":
    main()
