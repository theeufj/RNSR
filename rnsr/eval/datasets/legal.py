"""Legal benchmarks: CUAD, ContractNLI, LegalBench.

Shapes differ, and honesty about fit matters for interpretation:
  CUAD          clause extraction from full contracts — the DocDB regime
                (needles in long documents; empty answers are real cases).
  ContractNLI   premise/hypothesis NLI at clause scale (this public mirror
                is span-level, not whole-NDA) — short reasoning.
  LegalBench    162 short legal-reasoning tasks — tests the harness
                generically, not the retrieval thesis. Allowed labels are
                derived from each task's train split and quoted in the
                question.
"""

from __future__ import annotations

import random

from rnsr.eval.datasets.base import EvalItem

CUAD_ID = "theatticusproject/cuad-qa"
CONTRACTNLI_ID = "presencesw/contractnli"
LEGALBENCH_ID = "nguha/legalbench"

# A representative, buyer-relevant slice of LegalBench's 162 tasks:
# contract/consumer/privacy QA plus one classic rule-application task.
LEGALBENCH_DEFAULT_TASKS = (
    "consumer_contracts_qa",
    "privacy_policy_qa",
    "contract_nli_explicit_identification",
    "abercrombie",
)


def load_cuad(limit: int | None = 30, *, seed: int = 11,
              max_context_chars: int = 400_000) -> list[EvalItem]:
    """Clause-extraction questions grouped over few contracts (each distinct
    contract becomes one ingested corpus, so grouping controls ingest cost).
    Keeps the natural mix of present and absent clauses."""
    from datasets import load_dataset

    ds = load_dataset(CUAD_ID, revision="refs/convert/parquet", split="test")
    by_contract: dict[str, list] = {}
    for r in ds:
        if len(r["context"]) <= max_context_chars:
            by_contract.setdefault(r["title"], []).append(r)

    rng = random.Random(seed)
    contracts = sorted(by_contract)
    rng.shuffle(contracts)

    items: list[EvalItem] = []
    for title in contracts:
        rows = by_contract[title]
        rng.shuffle(rows)
        for r in rows[:6]:  # a handful per contract, then move on
            if limit and len(items) >= limit:
                return items
            answers = r["answers"]["text"]
            gold = " | ".join(answers) if answers else "No such clause"
            items.append(EvalItem(
                qid=f"cuad-{r['id'][:60]}",
                question=(
                    r["question"]
                    + " If the contract contains no such clause, answer "
                      "exactly: No such clause."
                ),
                gold=gold,
                task_class="extraction" if answers else "absent-clause",
                context=r["context"],
                meta={"contract": title, "n_answers": len(answers)},
            ))
    return items


def load_contractnli(limit: int | None = 30, *, seed: int = 11) -> list[EvalItem]:
    from datasets import load_dataset

    ds = load_dataset(CONTRACTNLI_ID, split="test")
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    items = []
    for i, r in enumerate(rows[: limit or len(rows)]):
        items.append(EvalItem(
            qid=f"contractnli-{i}",
            question=(
                "Premise (from an NDA):\n" + r["sentence1"] + "\n\n"
                "Hypothesis: " + r["sentence2"] + "\n\n"
                "Does the premise entail, contradict, or not address the "
                "hypothesis? Answer with exactly one word: entailment, "
                "contradiction, or neutral."
            ),
            gold=r["gold_label"],
            task_class="nli",
        ))
    return items


def load_legalbench(tasks: tuple[str, ...] = LEGALBENCH_DEFAULT_TASKS,
                    limit: int | None = 30, *, seed: int = 11) -> list[EvalItem]:
    from datasets import load_dataset

    rng = random.Random(seed)
    per_task = max(1, (limit or 32) // len(tasks))
    items: list[EvalItem] = []
    for task in tasks:
        train = load_dataset(LEGALBENCH_ID, task, split="train")
        test = load_dataset(LEGALBENCH_ID, task, split="test")
        labels = sorted({str(r["answer"]) for r in train})
        rows = list(test)
        rng.shuffle(rows)
        for i, r in enumerate(rows[:per_task]):
            fields = "\n".join(
                f"{k}: {v}" for k, v in r.items()
                if k not in ("answer", "index") and str(v).strip()
            )
            question = (
                f"LegalBench task '{task}'.\n{fields}\n\n"
                f"Answer with exactly one of: {', '.join(labels)}."
            )
            items.append(EvalItem(
                qid=f"legalbench-{task}-{i}",
                question=question,
                gold=str(r["answer"]),
                task_class=f"legalbench:{task}",
            ))
    return items[: limit or len(items)]
