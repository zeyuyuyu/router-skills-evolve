import math

from training.data.domain_rebalance import compute_domain_weights


def test_uniform_when_already_balanced():
    counts = {"airline": 100, "retail": 100, "telecom": 100}
    weights = compute_domain_weights(counts, temperature=0.5)
    assert math.isclose(weights["airline"], 1/3, abs_tol=1e-6)
    assert math.isclose(weights["retail"], 1/3, abs_tol=1e-6)
    assert math.isclose(weights["telecom"], 1/3, abs_tol=1e-6)


def test_temperature_zero_uniform():
    """T=0 -> uniform sampling regardless of counts."""
    counts = {"airline": 100, "retail": 200, "telecom": 700}
    weights = compute_domain_weights(counts, temperature=0.0)
    for v in weights.values():
        assert math.isclose(v, 1/3, abs_tol=1e-6)


def test_temperature_one_proportional():
    """T=1 -> exactly proportional to source counts."""
    counts = {"airline": 100, "retail": 200, "telecom": 700}
    weights = compute_domain_weights(counts, temperature=1.0)
    total = sum(counts.values())
    assert math.isclose(weights["airline"], 100/total, abs_tol=1e-6)
    assert math.isclose(weights["retail"], 200/total, abs_tol=1e-6)
    assert math.isclose(weights["telecom"], 700/total, abs_tol=1e-6)


def test_temperature_half_real_distribution():
    """The actual production case: telecom 60%, retail 21%, airline 20%."""
    counts = {"airline": 1254, "retail": 1342, "telecom": 3817}
    weights = compute_domain_weights(counts, temperature=0.5)
    assert weights["telecom"] < 0.50, f"Expected telecom < 50%, got {weights['telecom']:.3f}"
    assert weights["airline"] > 0.20, f"Expected airline > 20%, got {weights['airline']:.3f}"
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6)


def test_per_row_weight_via_division():
    """Each row's weight = domain_weight / row_count_in_domain (so sampling
    one row from a domain at probability domain_weight is uniform within the domain)."""
    from training.data.domain_rebalance import compute_per_row_weights
    rows = [
        {"_p": {"domain": "airline"}},
        {"_p": {"domain": "airline"}},
        {"_p": {"domain": "telecom"}},
    ]
    per_row = compute_per_row_weights(rows, temperature=0.5)
    assert len(per_row) == 3
    assert math.isclose(per_row[0], per_row[1], abs_tol=1e-6)
    counts = {"airline": 2, "telecom": 1}
    domain_weights = compute_domain_weights(counts, temperature=0.5)
    assert math.isclose(per_row[0] * 2, domain_weights["airline"], abs_tol=1e-6)
