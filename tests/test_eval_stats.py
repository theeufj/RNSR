"""Uncertainty helpers: intervals must widen with small n and the paired test
must count only discordant pairs."""
from rnsr.eval.stats import bootstrap_diff, describe, mcnemar, repeat_range, wilson


def test_wilson_interval_widens_with_small_n_and_stays_in_range():
    lo40, hi40 = wilson(34, 40)
    lo400, hi400 = wilson(340, 400)
    assert 0 <= lo40 < 0.85 < hi40 <= 1
    assert (hi40 - lo40) > 2 * (hi400 - lo400)
    assert wilson(0, 10)[0] == 0.0 and abs(wilson(10, 10)[1] - 1.0) < 1e-9
    assert wilson(0, 0) == (0.0, 0.0)


def test_mcnemar_uses_discordant_pairs_only():
    a = [True] * 30 + [False] * 10
    b = [True] * 30 + [False] * 10
    same = mcnemar(a, b)
    assert same["a_only"] == same["b_only"] == 0 and same["p_value"] == 1.0 and same["diff"] == 0.0
    b2 = [True] * 25 + [False] * 15          # a right on 5 where b wrong
    t = mcnemar(a, b2)
    assert t["a_only"] == 5 and t["b_only"] == 0 and abs(t["diff"] - 0.125) < 1e-9
    assert abs(t["p_value"] - 2 * (1 / 32)) < 1e-9   # exact binomial, 5 of 5 one way
    lo, hi = bootstrap_diff(a, b2, iters=2000)
    assert lo >= 0.0 and hi > 0.1


def test_repeat_range_and_describe():
    assert repeat_range([34, 36, 35]) == {"repeats": 3, "min": 34, "mean": 35.0, "max": 36}
    assert repeat_range([])["repeats"] == 0
    assert describe(34, 40).startswith("34/40 (85.0%, 95% CI ")
