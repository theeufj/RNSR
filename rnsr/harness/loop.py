"""The depth-1 RLM root loop (spec §4, §7).

prompt -> root code cell -> sandboxed exec -> observation -> repeat, until
FINAL/FINAL_VAR or a budget cap. The damping rule and the variable-recovery
fallback are harness mechanics, not prompt requests — they fire regardless
of what the model does.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from rnsr.config import Settings
from rnsr.env.sandbox import SandboxedRepl
from rnsr.errors import SandboxError
from rnsr.harness.budget import BudgetLedger
from rnsr.harness.prompts.base import render_system, render_transcript
from rnsr.harness.recovery import recover_variable
from rnsr.harness.trajectory import TrajectoryWriter
from rnsr.llm.base import LLMClient
from rnsr.llm.batch import map_prompts

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_OBSERVATION_LIMIT = 4000


@dataclass
class EnvSpec:
    """What gets preloaded into the sandbox."""

    mode: str                       # 'classic' | 'docdb'
    context: str | None = None      # classic: the flat string
    corpus_db: str | None = None    # docdb: artifact path
    manifest: dict | None = None    # docdb: rendered into the system prompt


@dataclass
class QueryResult:
    answer: object
    status: str                     # 'final' | 'recovered' | 'budget_exhausted' | 'error'
    final: dict | None
    ledger: dict
    trajectory_path: str
    iterations: int
    breached_cap: str | None = None


@dataclass
class RootRunner:
    """Binds the root/sub clients + settings; run() executes one query."""

    root_client: LLMClient
    root_model: str
    sub_client: LLMClient
    sub_model: str
    settings: Settings = field(default_factory=Settings)
    embed_client: LLMClient | None = None   # enables search ladder rung 4
    embed_model: str = ""

    def _extract_code(self, text: str) -> str | None:
        blocks = _CODE_BLOCK.findall(text)
        if blocks:
            return "\n\n".join(b.strip() for b in blocks)
        # a bare-code reply (no fence) still gets executed if it looks like code
        stripped = text.strip()
        if stripped and not stripped.split()[0][0].isupper():
            return stripped
        return None

    def _rpc_handlers(self, ledger: BudgetLedger, trajectory: TrajectoryWriter) -> dict:
        async def llm_batch(request: dict) -> dict:
            prompts = request["prompts"]
            remaining = ledger.max_sub_calls - ledger.sub_calls
            if len(prompts) > remaining:
                raise RuntimeError(
                    f"sub-call budget exceeded: batch of {len(prompts)} > "
                    f"{remaining} remaining"
                )
            responses = await map_prompts(
                self.sub_client, prompts, model=self.sub_model,
                concurrency=self.settings.sub_concurrency,
                on_usage=lambda u: ledger.add_usage(u, sub_call=True),
            )
            trajectory.event("sub_batch", n=len(prompts),
                             failed=sum(r is None for r in responses))
            return {"results": [r.text if r else "" for r in responses]}

        async def log(request: dict) -> dict:
            data = {k: v for k, v in request.items() if k not in ("kind", "op", "event")}
            trajectory.event(request.get("event", "env_log"), **data)
            return {}

        async def embed(request: dict) -> dict:
            if self.embed_client is None:
                raise RuntimeError("no embed client configured (rung 4 dormant)")
            vectors = await self.embed_client.embed(request["texts"],
                                                    model=self.embed_model)
            trajectory.event("embed_batch", n=len(request["texts"]))
            return {"vectors": vectors}

        return {"llm_batch": llm_batch, "log": log, "embed": embed}

    async def run(self, question: str, env: EnvSpec, *,
                  run_dir: str | Path | None = None,
                  query_id: str | None = None) -> QueryResult:
        s = self.settings
        ledger = BudgetLedger.from_settings(s)
        query_id = query_id or uuid.uuid4().hex[:12]
        trajectory = TrajectoryWriter(run_dir or s.run_dir, query_id)
        trajectory.event("start", question=question, mode=env.mode,
                         root_model=self.root_model, sub_model=self.sub_model)

        system = render_system(env.mode, manifest=env.manifest,
                               batch_chars=s.sub_call_char_budget,
                               provider=getattr(self.root_client, "provider", ""))
        sandbox = SandboxedRepl(rpc_handlers=self._rpc_handlers(ledger, trajectory))
        turns: list[tuple[str, str]] = []
        final: dict | None = None
        seen_candidates: dict[str, int] = {}
        damped = False
        completeness_checked = False
        budget_warned = False

        try:
            await sandbox.start(mode=env.mode, context=env.context,
                                corpus_db=env.corpus_db)
            while final is None:
                cap = ledger.breached()
                if cap:
                    trajectory.event("budget_breached", cap=cap, **ledger.snapshot())
                    result = await recover_variable(
                        sandbox, self, question, turns, trajectory
                    )
                    return self._finish(result, "recovered" if result else "budget_exhausted",
                                        ledger, trajectory, turns, breached=cap)

                prompt = render_transcript(question, turns)
                resp = await self._root_complete(prompt, system, ledger, trajectory)
                if resp is None:   # provider unreachable — salvage what exists
                    trajectory.event("budget_breached", cap="root_timeout",
                                     **ledger.snapshot())
                    result = await recover_variable(
                        sandbox, self, question, turns, trajectory
                    )
                    return self._finish(result,
                                        "recovered" if result else "budget_exhausted",
                                        ledger, trajectory, turns,
                                        breached="root_timeout")
                ledger.add_usage(resp.usage)
                ledger.root_iters += 1
                code = self._extract_code(resp.text)
                if code is None:
                    turns.append(("# (no code block found in your reply)",
                                  "Reply with exactly one ```python code block."))
                    trajectory.event("no_code", reply=resp.text[:500])
                    continue

                try:
                    cell = await sandbox.exec_cell(
                        code, timeout=min(s.cell_timeout_s, ledger.remaining_wall_s())
                    )
                except SandboxError as e:
                    # A runaway cell killed the sandbox (seen live: 120s
                    # cell → whole query lost). Restart it — preloads are
                    # reconstructable; only user variables are lost — and
                    # let the loop continue.
                    trajectory.event("sandbox_restarted", error=str(e)[:200])
                    await sandbox.start(mode=env.mode, context=env.context,
                                        corpus_db=env.corpus_db)
                    turns.append((code, (
                        f"[harness] {e} The sandbox was restarted: db/doc/"
                        "manifest and tools are reloaded, but YOUR VARIABLES "
                        "ARE GONE. Recompute what you need with cheaper "
                        "operations (avoid full-text scans in pure Python; "
                        "use search()/SQL/regex instead)."
                    )))
                    continue
                observation = self._observe(cell)
                trajectory.event("cell", code=code, ok=cell.ok,
                                 stdout=cell.stdout[:2000], error=cell.error,
                                 final=cell.final, rpc_count=cell.rpc_count)

                if cell.final is not None:
                    if not completeness_checked:
                        completeness_checked = True
                        gap = await self._completeness_gap(
                            question, cell.final, ledger)
                        if gap:
                            trajectory.event("completeness_pushback", gap=gap)
                            turns.append((code, (
                                "[harness] FINAL not accepted yet — the answer "
                                f"seems incomplete: {gap} Address this and "
                                "call FINAL again (or resubmit unchanged if "
                                "you believe it is complete)."
                            )))
                            continue
                    final = cell.final
                    break

                # Budget pressure (§7): the harness can see the wall clock;
                # the model can't. Seen live: five careful exploration turns,
                # then death mid-thought with the right verdict unconcluded.
                remaining = ledger.max_wall_s - ledger.wall_s
                iters_left = ledger.max_root_iters - ledger.root_iters
                if not budget_warned and (remaining < max(120.0, 0.2 * ledger.max_wall_s)
                                          or iters_left <= 2):
                    budget_warned = True
                    observation += (
                        f"\n[harness] BUDGET LOW: ~{int(remaining)}s and "
                        f"{iters_left} iterations remain. Converge NOW: give "
                        "FINAL with the best-supported answer from what you "
                        "have already seen (including a definitive negative "
                        "like 'No such clause' if that is where the evidence "
                        "points). Do not start new exploration."
                    )
                    trajectory.event("budget_warning", remaining_s=int(remaining),
                                     iters_left=iters_left)

                # Damping (§7): same normalized output recomputed twice ->
                # force a confirm-or-reject turn, once.
                key = _normalize(cell.stdout)
                if key and cell.ok:
                    seen_candidates[key] = seen_candidates.get(key, 0) + 1
                    if seen_candidates[key] >= 2 and not damped:
                        damped = True
                        observation += (
                            "\n[harness] You have computed this same result before. "
                            "Either call FINAL(...)/FINAL_VAR(...) with it now, or "
                            "state in a comment what is still unverified and check "
                            "that one thing."
                        )
                        trajectory.event("damping", value=key[:200])
                turns.append((code, observation))

            trajectory.event("final", **final)
            return self._finish(final, "final", ledger, trajectory, turns)
        except Exception as e:
            trajectory.event("error", error=f"{type(e).__name__}: {e}")
            return self._finish(None, "error", ledger, trajectory, turns)
        finally:
            await sandbox.close()
            trajectory.close()

    async def _root_complete(self, prompt: str, system: str,
                             ledger: BudgetLedger, trajectory) -> object | None:
        """Root call with a harness-side timeout tied to the wall budget.

        Provider SDK defaults allow requests to hang for up to 10 minutes —
        long enough for one stuck call to eat the entire §7 wall cap (seen
        live on FinanceBench). Three attempts with backoff (transient
        network loss killed back-to-back attempts, also seen live), each
        capped, then give up so recovery still has budget to run.
        """
        for attempt in (1, 2, 3):
            timeout = min(120.0, ledger.remaining_wall_s())
            try:
                async with asyncio.timeout(timeout):
                    return await self.root_client.complete(
                        prompt, model=self.root_model, system=system,
                        max_tokens=8192, seed=self.settings.llm_seed,
                    )
            except Exception as e:  # timeout or any provider error
                trajectory.event("root_call_failed", attempt=attempt,
                                 timeout_s=timeout, error=f"{type(e).__name__}: {e}"[:200])
                backoff = min(5.0 * attempt, ledger.remaining_wall_s() / 4)
                if backoff > 0.1:
                    await asyncio.sleep(backoff)
        return None

    async def _completeness_gap(self, question: str, final: dict,
                                ledger: BudgetLedger) -> str | None:
        """One cheap sub-LM check: does the draft answer address every part
        of the question? Returns the gap description, or None to accept.
        Anything but an explicit MISSING verdict accepts — this is a nudge
        against dropped question parts (seen live), not a second judge."""
        prompt = (
            f"Question: {question}\n\n"
            f"Draft answer: {str(final.get('value'))[:1500]}\n\n"
            "Does the draft answer address EVERY quantity and part the "
            "question asks for (names, magnitudes, all requested "
            "components)? Judge coverage only, not correctness. Reply with "
            "exactly COMPLETE, or 'MISSING: <what is missing>' in one line."
        )
        try:
            resp = await self.sub_client.complete(prompt, model=self.sub_model,
                                                  max_tokens=100)
            ledger.add_usage(resp.usage, sub_call=True)
        except Exception:
            return None
        text = resp.text.strip()
        if text.upper().startswith("MISSING"):
            return text[len("MISSING"):].lstrip(": ").strip() or "unspecified gap"
        return None

    def _observe(self, cell) -> str:
        text = cell.stdout if cell.ok else (cell.stdout + (cell.error or ""))
        if len(text) > _OBSERVATION_LIMIT:
            text = (text[:_OBSERVATION_LIMIT]
                    + f"\n…[truncated {len(text) - _OBSERVATION_LIMIT} chars]")
        return text or "(no output)"

    def _finish(self, final: dict | None, status: str, ledger: BudgetLedger,
                trajectory: TrajectoryWriter, turns: list,
                breached: str | None = None) -> QueryResult:
        trajectory.event("end", status=status, **ledger.snapshot())
        return QueryResult(
            answer=final.get("value") if final else None,
            status=status,
            final=final,
            ledger=ledger.snapshot(),
            trajectory_path=str(trajectory.path),
            iterations=ledger.root_iters,
            breached_cap=breached,
        )


def _normalize(text: str) -> str:
    out = re.sub(r"\s+", " ", text.strip().lower())
    return out[:300]
