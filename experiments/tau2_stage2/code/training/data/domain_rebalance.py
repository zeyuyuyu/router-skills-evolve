"""Domain rebalance via temperature sampling (Spec §5.4).

Source distribution: telecom 60%, retail 21%, airline 20%.
At T=0.5, effective sampling becomes ~telecom 46%, retail 27%, airline 27%
(weighted by sqrt of source count, then normalized).

Reference: arxiv 2410.04579 — upsampling outperforms per-token loss weighting
for SFT stability.
"""
from __future__ import annotations


def compute_domain_weights(counts: dict[str, int], temperature: float) -> dict[str, float]:
    """Compute domain-level sampling probabilities.

    weight(domain) ∝ count(domain) ** temperature

    Args:
        counts: {domain: row_count}
        temperature: 0.0 → uniform; 1.0 → proportional; 0.5 → in between.

    Returns:
        Normalized {domain: probability} summing to 1.0.
    """
    if not counts:
        return {}
    if temperature == 0.0:
        n = len(counts)
        return {d: 1.0 / n for d in counts}
    raw = {d: c ** temperature for d, c in counts.items()}
    total = sum(raw.values())
    return {d: v / total for d, v in raw.items()}


def compute_per_row_weights(rows: list[dict], temperature: float) -> list[float]:
    """Compute per-row sampling weights for use with WeightedRandomSampler.

    Each row's weight = domain_weight / row_count_in_domain so that:
    - P(sample from domain D) = domain_weight(D)
    - Within domain D, each row is uniform.

    Args:
        rows: list of dicts each with `_p.domain` field.
        temperature: see compute_domain_weights.

    Returns:
        List of per-row weights, same length as rows.
    """
    counts: dict[str, int] = {}
    for r in rows:
        d = r["_p"]["domain"]
        counts[d] = counts.get(d, 0) + 1
    domain_weights = compute_domain_weights(counts, temperature)
    return [domain_weights[r["_p"]["domain"]] / counts[r["_p"]["domain"]] for r in rows]
