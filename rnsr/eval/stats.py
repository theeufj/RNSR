"""Uncertainty for benchmark accuracies.

A single run on forty questions has a standard error of two to three
questions, which is the size of every arm difference this project has
reported. These helpers make that visible: a Wilson interval on each
accuracy, a paired exact McNemar test and bootstrap interval on each
difference (the arms answer identical questions, so the paired test is the
right one), and a per-repeat range when the same arm was run more than once.
Pure Python; no SciPy dependency.
"""
from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence


def wilson(correct: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n <= 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test p-value at p=0.5."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Exact McNemar test on paired correctness vectors over identical items.

    Returns the discordant counts (a right/b wrong, a wrong/b right), the
    two-sided p-value of the exact binomial on the discordant pairs, and the
    difference in accuracy."""
    if len(a) != len(b):
        raise ValueError("paired vectors must have the same length")
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = len(a)
    return {"n": n, "a_only": only_a, "b_only": only_b,
            "diff": (sum(a) - sum(b)) / n if n else 0.0,
            "p_value": _binom_two_sided(only_a, only_a + only_b)}


def bootstrap_diff(a: Sequence[bool], b: Sequence[bool], *, iters: int = 5000,
                   seed: int = 11, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval on accuracy(a) - accuracy(b), resampling
    items (paired) with replacement."""
    if len(a) != len(b) or not a:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(a)
    pairs = list(zip(a, b, strict=True))
    diffs = []
    for _ in range(iters):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append((sum(x for x, _ in sample) - sum(y for _, y in sample)) / n)
    diffs.sort()
    lo = diffs[int((1 - level) / 2 * (iters - 1))]
    hi = diffs[int((1 + level) / 2 * (iters - 1))]
    return (lo, hi)


def repeat_range(correct_counts: Iterable[int]) -> dict:
    """Min, mean and max correct across repeats of the same arm."""
    counts = list(correct_counts)
    if not counts:
        return {"repeats": 0, "min": None, "mean": None, "max": None}
    return {"repeats": len(counts), "min": min(counts),
            "mean": sum(counts) / len(counts), "max": max(counts)}


def describe(correct: int, n: int) -> str:
    """'34/40 (85.0%, 95% CI 71–93%)' for prose and tables."""
    lo, hi = wilson(correct, n)
    pct = correct / n * 100 if n else 0.0
    return f"{correct}/{n} ({pct:.1f}%, 95% CI {lo * 100:.0f}–{hi * 100:.0f}%)"
