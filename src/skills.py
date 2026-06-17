"""Skills 数据结构 + 学习算法

核心思想:
- 所有 prompt 共享单一 signature "coding"（全局一个 skill bucket）
- Skill 记录该 signature 下每个模型的 success/total 统计
- 基于统计决定"能不能用便宜模型"
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Prompt → Signature 提取
# ============================================================================

def extract_signature(prompt: str) -> str:  # noqa: ARG001
    """All tasks share a single global skill bucket."""
    return "coding"


# ============================================================================
# Procedure distillation (heuristic, no-API)
# ============================================================================

import re as _re

_CODE_BLOCK = _re.compile(r"```[\w+-]*\n(.*?)```", _re.DOTALL)
_TOOL_CALL = _re.compile(r"\b([a-z_][a-z0-9_]{2,})\s*\(")
_TAGGED_TOOL = _re.compile(r"<(?:tool|tool_call|function)>\s*([^<]+?)\s*</(?:tool|tool_call|function)>", _re.IGNORECASE)


def _heuristic_procedure(signature: str, exemplars: List[Dict], max_chars: int = 1200) -> str:
    """Induce a reusable scaffold from successful trajectories without any LLM.

    Pulls (a) reusable code snippets from fenced blocks, (b) tool-call / function
    names in call order, (c) a short domain hint, and assembles a compact
    "how to solve this cluster" procedure. This is the no-API floor; pass a
    distiller callable to Skill.distill_procedure for an LLM-quality version.
    """
    snippets: List[str] = []
    tools: List[str] = []
    for ex in exemplars:
        c = ex.get("completion", "") or ""
        for blk in _CODE_BLOCK.findall(c):
            blk = blk.strip()
            if blk and blk not in snippets:
                snippets.append(blk)
        for t in _TAGGED_TOOL.findall(c):
            t = t.strip().split()[0] if t.strip() else ""
            if t and t not in tools:
                tools.append(t)
        if not tools:  # fall back to bare function-call names
            for t in _TOOL_CALL.findall(c):
                if t not in ("if", "for", "while", "print", "return", "range") and t not in tools:
                    tools.append(t)

    lines = [f"# Procedure for cluster `{signature}`",
             f"# distilled from {len(exemplars)} successful exemplar(s)"]
    if tools:
        lines.append("")
        lines.append("Typical tool / call sequence: " + " -> ".join(tools[:12]))
    if snippets:
        lines.append("")
        lines.append("Reusable snippet:")
        lines.append("```")
        lines.append(snippets[0][:600])
        lines.append("```")
    if not tools and not snippets:
        # no code/tools: keep a trimmed exemplar completion as the scaffold
        sample = (exemplars[0].get("completion", "") or "").strip()
        if sample:
            lines.append("")
            lines.append("Reference solution shape:")
            lines.append(sample[:600])
    return "\n".join(lines)[:max_chars]


def make_llm_distiller(model_id: str, use_proxy: bool = False):
    """Return a (signature, exemplars) -> str callable backed by an LLM.

    Falls back silently to the heuristic if the API call fails, so the pipeline
    never hard-crashes on a missing key or rate-limit.

    Usage::
        distiller = make_llm_distiller("deepseek/deepseek-v3.2")
        sb.distill_all(distiller=distiller)
    """
    def _distiller(signature: str, exemplars: List[Dict],
                   prev_procedure: str = "") -> str:
        # lazy import to avoid pulling in openai at module load time
        try:
            from .models import call_llm  # noqa: PLC0415
        except ImportError:
            return _heuristic_procedure(signature, exemplars)

        # ── Incremental / map-reduce distillation over ALL exemplars ──────────
        # All solved tasks accumulate under the single global skill, so distil
        # from EVERY one, in batches of ≤ SKILL_DISTILL_BATCH (default 100). The
        # procedure is carried forward: batch 1 distils, batch 2..N refine the
        # running procedure with their examples. If a batch overflows the model
        # context, its size is halved and the chunk retried. Each example is
        # truncated to bound tokens.
        batch_size = max(1, int(os.environ.get("SKILL_DISTILL_BATCH", "100")))

        def _fmt(batch: List[Dict], start: int) -> str:
            blocks = []
            for j, ex in enumerate(batch, start):
                p = (ex.get("prompt") or "").strip()[:400]
                c = (ex.get("completion") or "").strip()[:500]
                blocks.append(
                    f"### Example {j}  (task_id={ex.get('task_id', '?')})\n"
                    f"**Problem**:\n```python\n{p}\n```\n"
                    f"**Solution**:\n```python\n{c}\n```"
                )
            return "\n\n".join(blocks)

        def _prompt(examples_text: str, prev: str) -> str:
            refine = bool((prev or "").strip())
            system = (
                "You are an expert coding-skills curator. "
                "Given a cluster of similar coding problems and their solutions"
                + (", plus the CURRENT cheatsheet so far, improve it"
                   if refine else ", write a tight CHEATSHEET")
                + " that helps a SMALL (1.5B) model solve these faster. "
                + ("Keep what works, fix gaps the new examples reveal, prune fluff. "
                   if refine else "")
                + "Be CONCRETE and SHORT — a small model gains nothing from generic "
                "advice it already knows (e.g. 'use all/any/sum', 'handle edge cases'). "
                "Give only HIGH-SIGNAL, transferable content:\n"
                "1. **Recipes** — 2-4 copy-pasteable code idioms that recur across these "
                "problems (each ≤4 lines, with a one-line 'use when ...' trigger)\n"
                "2. **Gotchas** — up to 3 SPECIFIC mistakes seen in these solutions "
                "(off-by-one, order/dup loss, wrong return shape — not platitudes)\n"
                "Plain Markdown, no preamble, NO step-by-step prose, NO restating the "
                "obvious. Hard cap 150 words. If there is no transferable pattern, say so "
                "in one line rather than padding."
            )
            prev_block = (
                "## CURRENT procedure so far (refine this)\n" + prev.strip()[:2000]
                + "\n\n---\n" if refine else ""
            )
            user = (
                f"Cluster signature: `{signature}`\n\n" + prev_block
                + "## Solved examples\n" + examples_text
                + ("\n\n---\nNow write the IMPROVED Procedure." if refine
                   else "\n\n---\nNow write the Procedure.")
            )
            return f"{system}\n\n{user}"

        def _is_ctx_err(msg: str) -> bool:
            m = (msg or "").lower()
            return any(k in m for k in
                       ("context", "maximum", "too long", "max_tokens", "length",
                        "token limit", "exceed"))

        def _call(prompt: str):
            try:
                r = call_llm(model_id=model_id, prompt=prompt, use_proxy=use_proxy,
                             temperature=0.3, max_tokens=400)
            except Exception as e:  # noqa: BLE001 — missing key / network
                return None, str(e)
            return (r.get("response") or "").strip() or None, r.get("error")

        def _distill_batch(args) -> Optional[str]:
            """MAP: distil ONE batch independently; halve & retry on ctx overflow."""
            start, batch = args
            cur = len(batch)
            while True:
                sub = batch[:cur]
                resp, err = _call(_prompt(_fmt(sub, start), ""))
                if resp:
                    return resp
                if _is_ctx_err(err) and cur > 1:
                    cur = max(1, cur // 2)
                    continue
                return None

        # Split ALL exemplars into ≤batch_size chunks.
        batches = [(i + 1, exemplars[i:i + batch_size])
                   for i in range(0, len(exemplars), batch_size)]

        # MAP — distil batches concurrently (CommonStack API handles concurrency).
        if len(batches) <= 1:
            partials = [_distill_batch(batches[0])] if batches else []
        else:
            workers = min(int(os.environ.get("SKILL_DISTILL_WORKERS", "8")), len(batches))
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:
                partials = list(pool.map(_distill_batch, batches))
        partials = [p for p in partials if p]

        if not partials:
            return _heuristic_procedure(signature, exemplars)

        # REDUCE — merge the partial procedures (+ previous cycle's, if any) into
        # one. A single partial with no prior procedure needs no merge.
        prev = (prev_procedure or "").strip()
        if len(partials) == 1 and not prev:
            procedure = partials[0]
        else:
            sources = ([("Previous-cycle procedure", prev)] if prev else []) + \
                      [(f"Partial procedure {i+1}", p) for i, p in enumerate(partials)]
            merge_sys = (
                "Merge these partial coding CHEATSHEETS (same cluster) into ONE, "
                "plain Markdown, no preamble: a **Recipes** list (copy-pasteable "
                "idioms, each with a 'use when ...' trigger) and a **Gotchas** list "
                "(specific mistakes). Deduplicate, keep only high-signal transferable "
                "content, NO generic advice, NO step-by-step prose. Hard cap 150 words."
            )
            merge_user = f"Cluster signature: `{signature}`\n\n" + "\n\n---\n".join(
                f"## {name}\n{text.strip()[:1500]}" for name, text in sources
            ) + "\n\n---\nNow output the single merged cheatsheet."
            resp, _err = _call(f"{merge_sys}\n\n{merge_user}")
            procedure = resp or max(partials, key=len)   # fall back to longest partial

        header = (
            f"# Coding cheatsheet `{signature}`\n"
            f"# distilled by {model_id} from {len(exemplars)} exemplar(s) "
            f"(parallel map-reduce, {len(batches)} batch(es)≤{batch_size})\n\n"
        )
        return (header + procedure)[:1400]

    return _distiller


# ============================================================================
# Skill (一个 cluster 的统计)
# ============================================================================

class Skill:
    """
    对一个 signature cluster 记录每个模型的成功率统计。
    
    核心方法:
    - update(model, success): 记一笔新数据
    - model_success_rate(model): 返回该 model 对这类题的成功率
    - recommend_cheapest_viable_model(candidates, min_rate): 推荐最便宜可用模型
    """

    # With a single global skill, ALL solved tasks accumulate here. Keep them all
    # (distillation now batches over everything via map-reduce); env override for
    # memory-constrained runs. (Was 8 — far too few, wasted most traces.)
    MAX_EXEMPLARS = int(os.environ.get("SKILL_MAX_EXEMPLARS", "10000"))

    def __init__(self, signature: str):
        self.signature = signature
        # {model_id: [successes, total]}
        self.stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        self.history: List[Dict] = []  # 完整事件记录
        # --- procedural skill fields (review item 2, 2026-05-21) ---
        # A skill is no longer just a success-rate counter. It also accumulates
        # successful trajectories (exemplars) and a distilled, reusable
        # "procedure" (how to solve this cluster: tool-use steps, domain policy,
        # reusable code snippets, agent sub-workflow). The stats above still
        # drive routing; these fields make the skill teachable to the small
        # model and inspectable by a human.
        self.exemplars: List[Dict] = []   # [{task_id, prompt, completion, model}]
        self.procedure: str = ""          # distilled reusable scaffold
        self.procedure_source: str = ""   # "" | "heuristic" | "llm"

    def update(self, model_id: str, success: bool, task_id: str = ""):
        """记录一次调用结果 (stats only — exemplars via add_exemplar)"""
        self.stats[model_id][1] += 1
        if success:
            self.stats[model_id][0] += 1
        self.history.append({
            "task_id": task_id,
            "model": model_id,
            "success": success,
        })

    def add_exemplar(self, prompt: str, completion: str, model_id: str,
                     task_id: str = "") -> None:
        """Store a successful trajectory for this cluster (capped, most-recent)."""
        if not prompt or not completion:
            return
        self.exemplars.append({
            "task_id": task_id,
            "prompt": prompt,
            "completion": completion,
            "model": model_id,
        })
        if len(self.exemplars) > self.MAX_EXEMPLARS:
            self.exemplars = self.exemplars[-self.MAX_EXEMPLARS:]

    def distill_procedure(self, distiller=None) -> str:
        """Induce a reusable procedure from the accumulated exemplars.

        distiller: optional callable (signature, exemplars) -> str. When given
        (e.g. an LLM-backed summariser, the "agent sub-workflow" path), it
        produces the procedure. When None, a no-API heuristic extracts reusable
        code snippets + tool-call patterns + a domain hint from the exemplars.
        """
        if not self.exemplars:
            return self.procedure
        if distiller is not None:
            # Pass the PREVIOUS procedure so the distiller refines it (true skill
            # evolution) rather than regenerating from scratch each cycle. Older
            # distillers that take only (signature, exemplars) still work.
            try:
                self.procedure = distiller(self.signature, self.exemplars,
                                           self.procedure)
            except TypeError:
                self.procedure = distiller(self.signature, self.exemplars)
            self.procedure_source = "llm"
        else:
            self.procedure = _heuristic_procedure(self.signature, self.exemplars)
            self.procedure_source = "heuristic"
        return self.procedure

    def model_success_rate(
        self, model_id: str, use_laplace: bool = True
    ) -> Tuple[float, int]:
        """
        返回 (success_rate, n_samples)。
        use_laplace=True 时用 Laplace 平滑 (避免小样本过估)。
        """
        s, t = self.stats[model_id]
        if t == 0:
            return 0.5 if use_laplace else 0.0, 0
        if use_laplace:
            return (s + 1) / (t + 2), t
        return s / t, t

    def recommend_cheapest_viable_model(
        self,
        candidates: List[str],  # 按价格从便宜到贵排序
        min_rate: float = 0.85,
        min_samples: int = 2,
    ) -> Optional[str]:
        """
        推荐最便宜但成功率 >= min_rate 的模型。
        
        Args:
            candidates: 模型 ID 列表 (按价格从便宜到贵排)
            min_rate: 最低可接受成功率 (Laplace 平滑后)
            min_samples: 至少要几个样本才能做推荐
        
        Returns:
            推荐的 model_id, 或 None (不确定)
        """
        for mid in candidates:
            rate, n = self.model_success_rate(mid, use_laplace=True)
            if n >= min_samples and rate >= min_rate:
                return mid
        return None

    def can_downgrade_to_small(
        self,
        small_model: str,
        min_rate: float = 0.8,
        min_samples: int = 1,
    ) -> Optional[bool]:
        """
        判断这个 cluster 能否降级到小模型？
        
        Returns:
            True  : 小模型能胜任 (省钱)
            False : 小模型不行, 必须升级
            None  : 数据不够, 无法判断
        """
        s, t = self.stats[small_model]
        if t < min_samples:
            return None
        return (s / t) >= min_rate

    def to_dict(self) -> Dict:
        """序列化为 JSON"""
        return {
            "signature": self.signature,
            "stats": {mid: list(v) for mid, v in self.stats.items()},
            "history": self.history,
            "exemplars": self.exemplars,
            "procedure": self.procedure,
            "procedure_source": self.procedure_source,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Skill":
        """从 JSON 反序列化 (tolerates old files without procedural fields)"""
        s = cls(data["signature"])
        s.stats = defaultdict(lambda: [0, 0])
        for mid, v in data["stats"].items():
            s.stats[mid] = list(v)
        s.history = data.get("history", [])
        s.exemplars = data.get("exemplars", [])
        s.procedure = data.get("procedure", "")
        s.procedure_source = data.get("procedure_source", "")
        return s


# ============================================================================
# SkillBook: 管理所有 skills
# ============================================================================

class SkillBook:
    """所有 skills 的容器 + 读写"""

    def __init__(self):
        self.skills: Dict[str, Skill] = {}

    def get_or_create(self, signature: str) -> Skill:
        """拿到 skill，不存在就新建"""
        if signature not in self.skills:
            self.skills[signature] = Skill(signature)
        return self.skills[signature]

    def update(self, prompt: str, model_id: str, success: bool, task_id: str = "",
               completion: str = ""):
        """便利方法: 从 prompt 抽 signature 再更新。

        completion: when the call SUCCEEDED and a completion is provided, the
        trajectory is also stored as an exemplar (feeds procedure distillation).
        Backward compatible — omitting it preserves the old stats-only behaviour.
        """
        sig = extract_signature(prompt)
        skill = self.get_or_create(sig)
        skill.update(model_id, success, task_id)
        if success and completion:
            skill.add_exemplar(prompt, completion, model_id, task_id)

    def get_procedure(self, prompt: str) -> str:
        """Return the distilled procedure for the prompt's cluster (or '')."""
        skill = self.skills.get(extract_signature(prompt))
        return skill.procedure if skill else ""

    def nearest_exemplars(self, prompt: str, k: int = 3,
                          exclude_task_id: str = "") -> List[Dict]:
        """Top-k solved exemplars most similar to `prompt` (TF-IDF cosine).

        Concrete worked examples help a small model more than an abstract
        procedure. Retrieval is over the cluster's accumulated exemplars; at
        held-out eval time these come from TRAIN traces (no leakage). Falls back
        to most-recent-k if scikit-learn is unavailable.
        """
        skill = self.skills.get(extract_signature(prompt))
        if not skill or not skill.exemplars:
            return []
        pool = [e for e in skill.exemplars
                if not (exclude_task_id and e.get("task_id") == exclude_task_id)
                and (e.get("prompt") and e.get("completion"))]
        if len(pool) <= k:
            return pool
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            corpus = [e["prompt"] for e in pool] + [prompt]
            tfidf = TfidfVectorizer().fit_transform(corpus)
            sims = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
            order = sims.argsort()[::-1][:k]
            return [pool[i] for i in order]
        except Exception:  # noqa: BLE001
            return pool[-k:]

    def get_context(self, prompt: str, fewshot_k: int = 0,
                    with_procedure: bool = True, exclude_task_id: str = "") -> str:
        """Full skill context = distilled procedure + k nearest worked examples.

        This is what the small model sees as its `procedure` prefix. fewshot_k=0
        → procedure only (legacy). The abstract cheatsheet teaches the pattern;
        the concrete examples show it applied to similar problems.
        """
        parts = []
        if with_procedure:
            proc = self.get_procedure(prompt)
            if proc:
                parts.append(proc)
        if fewshot_k > 0:
            exs = self.nearest_exemplars(prompt, k=fewshot_k, exclude_task_id=exclude_task_id)
            if exs:
                blocks = ["## Worked examples (similar solved problems — adapt, don't copy)"]
                for i, e in enumerate(exs, 1):
                    p = (e.get("prompt") or "").strip()[:500]
                    c = (e.get("completion") or "").strip()[:600]
                    blocks.append(f"### Example {i}\n```python\n{p}\n```\n"
                                  f"Solution:\n```python\n{c}\n```")
                parts.append("\n\n".join(blocks))
        return "\n\n".join(parts)

    def distill_all(self, distiller=None) -> int:
        """(Re)distill procedures for every skill that has exemplars.

        distiller: optional (signature, exemplars) -> str callable (e.g. an
        LLM-backed "agent sub-workflow" summariser). None => no-API heuristic.
        Returns the number of skills given a non-empty procedure.
        """
        n = 0
        for skill in self.skills.values():
            proc = skill.distill_procedure(distiller=distiller)
            if proc:
                n += 1
        return n

    def recommend(
        self,
        prompt: str,
        candidates: List[str],
        min_rate: float = 0.85,
        min_samples: int = 2,
    ) -> Tuple[Optional[str], Optional[Skill]]:
        """
        对一个新 prompt 给出推荐。
        
        Returns:
            (recommended_model_id, matched_skill)
        """
        sig = extract_signature(prompt)
        skill = self.skills.get(sig)
        if skill is None:
            return None, None
        rec = skill.recommend_cheapest_viable_model(candidates, min_rate, min_samples)
        return rec, skill

    def save(self, path: Path):
        """保存到 JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {"skills": [s.to_dict() for s in self.skills.values()]},
                f,
                indent=2,
            )

    def load(self, path: Path):
        """从 JSON 加载"""
        path = Path(path)
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        for s_data in data.get("skills", []):
            s = Skill.from_dict(s_data)
            self.skills[s.signature] = s

    def summary(self) -> Dict:
        """返回简要统计"""
        total = len(self.skills)
        return {
            "total_skills": total,
            "total_observations": sum(
                sum(v[1] for v in s.stats.values()) for s in self.skills.values()
            ),
            "signatures": list(self.skills.keys()),
        }
