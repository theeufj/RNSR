"""The depth-1 RLM root loop (spec §4, §7).

prompt -> root code cell -> sandboxed exec -> observation -> repeat, until
FINAL/FINAL_VAR or a budget cap. The damping rule and the variable-recovery
fallback are harness mechanics, not prompt requests — they fire regardless
of what the model does.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from rnsr.config import Settings
from rnsr.env.sandbox import SandboxedRepl
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
                resp = await self.root_client.complete(
                    prompt, model=self.root_model, system=system,
                    max_tokens=8192, seed=s.llm_seed,
                )
                ledger.add_usage(resp.usage)
                ledger.root_iters += 1
                code = self._extract_code(resp.text)
                if code is None:
                    turns.append(("# (no code block found in your reply)",
                                  "Reply with exactly one ```python code block."))
                    trajectory.event("no_code", reply=resp.text[:500])
                    continue

                cell = await sandbox.exec_cell(
                    code, timeout=min(120.0, ledger.remaining_wall_s())
                )
                observation = self._observe(cell)
                trajectory.event("cell", code=code, ok=cell.ok,
                                 stdout=cell.stdout[:2000], error=cell.error,
                                 final=cell.final, rpc_count=cell.rpc_count)

                if cell.final is not None:
                    final = cell.final
                    break

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
