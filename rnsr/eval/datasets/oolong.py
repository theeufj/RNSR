"""OOLONG loader (§8: linear-density long-context tasks; Phase B gate).

Real data loads from HuggingFace (id configurable — the public mirror has
moved before). For offline/CI use, `synthetic_oolong()` generates
classification-count tasks of the same shape with exactly known answers,
which is what the harness acceptance tests run against.
"""

from __future__ import annotations

import random

from rnsr.eval.datasets.base import EvalItem

DEFAULT_DATASET_ID = "oolong-bench/oolong"

_LABELS = ("numeric value", "entity", "location",
           "description and abstract concept", "abbreviation", "human being")

_TEMPLATES = {
    "numeric value": "User {u} asked: How many {thing}s were shipped in {year}?",
    "entity": "User {u} asked: Which company acquired {name} Corp?",
    "location": "User {u} asked: Where is the {name} facility located?",
    "description and abstract concept": "User {u} asked: What does resilience mean in {name} theory?",
    "abbreviation": "User {u} asked: What does {abbr} stand for?",
    "human being": "User {u} asked: Who founded the {name} institute?",
}


def load_oolong(dataset_id: str = DEFAULT_DATASET_ID, split: str = "test",
                limit: int | None = None) -> list[EvalItem]:
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split=split)
    except Exception as e:  # dataset moved/offline — fail with instructions
        raise RuntimeError(
            f"could not load OOLONG from '{dataset_id}': {e}\n"
            "Pass the correct HF dataset id (rnsr eval --benchmark oolong "
            "--dataset-id <id>) or use synthetic_oolong() for smoke tests."
        ) from e
    items = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        items.append(EvalItem(
            qid=f"oolong-{i}",
            question=row["question"],
            gold=str(row["answer"]),
            task_class=row.get("task", "oolong"),
            context=row["context"],
        ))
    return items


def synthetic_oolong(n_lines: int = 400, n_items: int = 5, seed: int = 7) -> list[EvalItem]:
    """OOLONG-shaped tasks: classify many lines, then answer an aggregate
    question whose gold answer is exactly countable."""
    rng = random.Random(seed)
    lines: list[tuple[int, str, str]] = []
    for u in range(n_lines):
        label = rng.choice(_LABELS)
        text = _TEMPLATES[label].format(
            u=u, thing=rng.choice(["widget", "gadget"]), year=rng.choice([2022, 2023]),
            name=rng.choice(["Acme", "Zenith", "Orion"]), abbr=rng.choice(["RLM", "FTS", "KPI"]),
        )
        lines.append((u, label, text))
    context = "\n".join(t for _, _, t in lines)

    items = []
    for i in range(n_items):
        label = _LABELS[i % len(_LABELS)]
        count = sum(1 for _, lab, _ in lines if lab == label)
        items.append(EvalItem(
            qid=f"syn-oolong-{i}",
            question=(
                "Each line of the context is a user question. Classify every "
                "line's expected answer type into exactly one of: "
                f"{', '.join(_LABELS)}. "
                f"How many lines are of type '{label}'? Answer with the count."
            ),
            gold=str(count),
            task_class="aggregation",
            context=context,
            meta={"label": label},
        ))
    return items
