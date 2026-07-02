"""Variable-recovery fallback (spec §4; mitigates the paper's B.2 failure,
where a correct answer was built in the namespace and then abandoned).

When the loop ends without a valid FINAL, inspect the REPL namespace for
answer-like variables and give the root LM one confirm-or-reject turn.
"""

from __future__ import annotations

import re

_ANSWERISH = re.compile(r"answer|result|final|out(put)?|pairs|total|value", re.IGNORECASE)


def rank_candidates(vars_: dict[str, dict]) -> list[str]:
    """Namespace vars, most answer-like first (name match, then recency —
    dict order reflects creation order, so later wins ties)."""
    names = list(vars_)
    return sorted(
        names,
        key=lambda n: (bool(_ANSWERISH.search(n)), names.index(n)),
        reverse=True,
    )


async def recover_variable(sandbox, runner, question: str, turns: list,
                           trajectory) -> dict | None:
    """One confirm-or-reject turn over ranked namespace candidates.

    Returns a FINAL-shaped dict ({"value", "encoding", "is_var"}) or None.
    """
    vars_ = await sandbox.vars()
    candidates = rank_candidates(vars_)[:8]
    if not candidates:
        return None

    listing = "\n".join(f"- {n}: {vars_[n]['type']} = {vars_[n]['repr']}"
                        for n in candidates)
    prompt = (
        f"TASK:\n{question}\n\n"
        "The session ended without FINAL being called. These variables were "
        "left in the namespace:\n"
        f"{listing}\n\n"
        "If one of them IS the answer to the task, reply with exactly its "
        "name. If none of them answers the task, reply with exactly NONE."
    )
    resp = await runner.root_client.complete(
        prompt, model=runner.root_model,
        system="You are confirming or rejecting a recovered answer. "
               "Reply with a single variable name or NONE.",
        max_tokens=64,
    )
    choice = resp.text.strip().strip("`'\"")
    trajectory.event("recovery", candidates=candidates, choice=choice)
    if choice == "NONE" or choice not in vars_:
        return None

    cell = await sandbox.exec_cell(f"FINAL_VAR({choice})", timeout=30.0)
    return cell.final
