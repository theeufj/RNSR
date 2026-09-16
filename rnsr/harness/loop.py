"""The depth-1 RLM root loop (spec §4, §7).

prompt -> root code cell -> sandboxed exec -> observation -> repeat, until
FINAL/FINAL_VAR or a budget cap. The damping rule and the variable-recovery
fallback are harness mechanics, not prompt requests — they fire regardless
of what the model does.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

from rnsr.answer_semantics import QueryStatus, Resolution, coerce_batch, comparison_key, is_negative
from rnsr.config import Settings
from rnsr.env.sandbox import SandboxedRepl
from rnsr.errors import SandboxError
from rnsr.harness.budget import BudgetLedger
from rnsr.harness.evidence import AnswerEvidence, from_final
from rnsr.harness.negative_audit import audit_negatives
from rnsr.harness.prompts.base import (
    render_batch_task,
    render_system,
    render_transcript,
)
from rnsr.harness.recovery import recover_variable
from rnsr.harness.trajectory import TrajectoryWriter
from rnsr.llm.base import LLMClient
from rnsr.llm.batch import map_prompts
from rnsr.obs import get_logger, log, metrics

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_OBSERVATION_LIMIT = 4000

_LOG = get_logger("harness.loop")


@dataclass
class EnvSpec:
    """What gets preloaded into the sandbox."""

    mode: str                       # 'classic' | 'docdb'
    context: str | None = None      # classic: the flat string
    corpus_db: str | None = None    # docdb: artifact path
    manifest: dict | None = None    # docdb: rendered into the system prompt
    playbook: object | None = None  # rnsr.harness.playbook.Playbook | None


@dataclass
class QueryResult:
    answer: object
    status: QueryStatus             # 'final' | 'recovered' | 'budget_exhausted' | 'error'
    final: dict | None
    ledger: dict
    trajectory_path: str
    iterations: int
    breached_cap: str | None = None
    health: dict | None = None          # corpus health snapshot, if gated
    evidence: AnswerEvidence | None = None
    raw_answer: object = None


@dataclass
class BatchQueryResult:
    """One batched loop's per-question answers.

    answers[qid] is the stripped answer text ("NOT_FOUND" when the model
    judged the corpus lacks it), or None when the loop never produced an
    answer for that qid — callers should fall back to a solo run for those.
    """

    answers: dict[str, str | None]
    result: QueryResult
    evidence: dict[str, AnswerEvidence] = field(default_factory=dict)


@dataclass
class ConsensusAnswer:
    """One field's answer after independent passes voted on it."""

    value: str | None
    resolved_by: Resolution     # 'unanimous' | 'majority' | 'tiebreak' | 'unresolved'
    agreement: float            # share of passes that produced the chosen value
    votes: list[str | None] = field(default_factory=list)
    evidence: AnswerEvidence | None = None

    @property
    def contested(self) -> bool:
        return self.resolved_by in ("split", "tiebreak", "unresolved") or self.value is None

    @property
    def tier(self) -> str | None:
        return self.evidence.tier if self.evidence else None


@dataclass
class ConsensusBatchResult:
    answers: dict[str, ConsensusAnswer]
    pass_results: list[QueryResult] = field(default_factory=list)
    tiebreak_results: dict[str, QueryResult] = field(default_factory=dict)

    @property
    def contested_qids(self) -> list[str]:
        return [q for q, a in self.answers.items() if a.contested]


def _vote_key(answer: str | None) -> str | None:
    """Comparison form for voting: two passes that wrote the same fact with
    different punctuation or casing must not read as a disagreement."""
    if answer is None:
        return None
    return comparison_key(answer) or None


def scale_budgets(s: Settings, n_questions: int) -> Settings:
    """Budget caps for an n-question batched loop.

    Each additional question adds half of a single question's caps: shared
    exploration amortizes the rest, and a full n× budget would let one
    confused batch burn n questions' worth of spend.
    """
    if n_questions <= 1:
        return s
    factor = 1.0 + 0.5 * (n_questions - 1)
    return replace(
        s,
        max_root_iters=round(s.max_root_iters * factor),
        max_sub_calls=round(s.max_sub_calls * factor),
        max_wall_s=s.max_wall_s * factor,
        max_spend_usd=s.max_spend_usd * factor,
    )




_coerce_batch = coerce_batch
_is_negative = is_negative  # historical import compatibility


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
        stripped = text.strip()
        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            return None
        # Bare prose can parse as an identifier or a quoted string. Execute
        # statements/calls, never those ambiguous expression-only replies.
        if not tree.body or any(isinstance(node, ast.Expr) and
                                isinstance(node.value, (ast.Name, ast.Constant))
                                for node in tree.body):
            return None
        return stripped

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
                on_usage=ledger.add_usage,
                on_attempt=ledger.reserve_sub_call,
                deadline=ledger._t0 + ledger.max_wall_s,
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
                  query_id: str | None = None,
                  batch_questions: list[tuple[str, str]] | None = None) -> QueryResult:
        batch_qids = [qid for qid, _ in batch_questions] if batch_questions else None
        s = self.settings
        ledger = BudgetLedger.from_settings(s)
        query_id = query_id or uuid.uuid4().hex[:12]
        trajectory = TrajectoryWriter(run_dir or s.run_dir, query_id,
                                      content=s.trajectory_content,
                                      key=s.trajectory_key)
        trajectory.event("start", question=question, mode=env.mode,
                         root_model=self.root_model, sub_model=self.sub_model)
        log(_LOG, logging.INFO, "query.start", query_id=query_id, mode=env.mode,
            root_model=self.root_model,
            batch_size=len(batch_qids) if batch_qids else 1)
        metrics().incr("queries_started", mode=env.mode)

        system = render_system(env.mode, manifest=env.manifest,
                               batch_chars=s.sub_call_char_budget,
                               provider=getattr(self.root_client, "provider", ""),
                               playbook=env.playbook)
        n_documents = len((env.manifest or {}).get("documents") or [])
        init_extra = {"enable_embeddings": self.embed_client is not None
                      and n_documents >= s.embed_auto_on_docs}
        sandbox = SandboxedRepl(rpc_handlers=self._rpc_handlers(ledger, trajectory),
                                fs_guard=s.sandbox_fs_guard)
        turns: list[tuple[str, str]] = []
        final: dict | None = None
        seen_candidates: dict[str, int] = {}
        damped = False
        completeness_checked = False
        negatives_audited = False
        budget_warned = False
        pushbacks = 0
        negative_audit = "none"
        health_grade = ((env.manifest or {}).get("health") or {}).get("grade")

        def _signals(**extra) -> AnswerEvidence:
            return from_final(
                extra.pop("final", final),
                status=extra.get("status", "final"),
                pushbacks=pushbacks,
                negative_audit=negative_audit,
                budget_warned=budget_warned,
                health_grade=health_grade,
                qid=extra.get("qid"),
            )

        try:
            await sandbox.start(mode=env.mode, context=env.context,
                                corpus_db=env.corpus_db, init_extra=init_extra)
            while final is None:
                cap = ledger.breached()
                if cap:
                    trajectory.event("budget_breached", cap=cap, **ledger.snapshot())
                    log(_LOG, logging.WARNING, "query.budget_breached",
                        query_id=query_id, cap=cap, **ledger.snapshot())
                    metrics().incr("budget_breaches", cap=cap)
                    result = await recover_variable(
                        sandbox, self, question, turns, trajectory, ledger
                    )
                    return self._finish(result, "recovered" if result else "budget_exhausted",
                                        ledger, trajectory, turns, breached=cap,
                                        evidence=_signals(final=result,
                                                          status="recovered" if result else "budget_exhausted"))

                final_hint = ("FINAL_BATCH({...}) with every question id"
                              if batch_qids else "FINAL(...)/FINAL_VAR(...)")
                prompt = render_transcript(question, turns, final_hint=final_hint)
                resp = await self._root_complete(prompt, system, ledger, trajectory)
                if resp is None:   # provider unreachable — salvage what exists
                    trajectory.event("budget_breached", cap="root_timeout",
                                     **ledger.snapshot())
                    result = await recover_variable(
                        sandbox, self, question, turns, trajectory, ledger
                    )
                    return self._finish(result,
                                        "recovered" if result else "budget_exhausted",
                                        ledger, trajectory, turns,
                                        breached="root_timeout",
                                        evidence=_signals(final=result,
                                                          status="recovered" if result else "budget_exhausted"))
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
                    log(_LOG, logging.WARNING, "sandbox.restarted",
                        query_id=query_id, error=str(e)[:200])
                    metrics().incr("sandbox_restarts")
                    await sandbox.start(mode=env.mode, context=env.context,
                                        corpus_db=env.corpus_db, init_extra=init_extra)
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
                    gap = None
                    if not completeness_checked:
                        completeness_checked = True
                        if batch_qids:
                            gap = self._batch_gap(cell.final, batch_qids)
                        else:
                            gap = await self._completeness_gap(
                                question, cell.final, ledger)
                    # Negative-answer audit (once, mechanical): a corpus
                    # probe for questions answered No/unknown/NOT_FOUND —
                    # lazy loops declare documented facts missing (seen
                    # live: verbatim values marked not-found after two
                    # shallow iterations). Solo negatives used to skip this.
                    if (gap is None and env.corpus_db and not negatives_audited):
                        negatives_audited = True
                        if batch_questions:
                            gap = audit_negatives(
                                cell.final, batch_questions, env.corpus_db,
                                trajectory)
                        else:
                            wrapped = {
                                "value": {query_id: cell.final.get("value")},
                                "verification": {
                                    query_id: cell.final.get("verification") or {},
                                },
                            }
                            gap = audit_negatives(
                                wrapped, [(query_id, question)], env.corpus_db,
                                trajectory)
                        negative_audit = "flagged" if gap else "probed"
                    elif gap is None and negative_audit == "flagged":
                        negative_audit = "survived"
                    if gap:
                        pushbacks += 1
                        trajectory.event("completeness_pushback", gap=gap)
                        log(_LOG, logging.INFO, "final.pushback",
                            query_id=query_id, batch_size=len(batch_qids or [query_id]))
                        metrics().incr("final_pushbacks")
                        fname = "FINAL_BATCH" if batch_qids else "FINAL"
                        turns.append((code, (
                            f"[harness] {fname} not accepted yet — the "
                            f"answer seems incomplete: {gap} Address this "
                            f"and call {fname} again (or resubmit "
                            "unchanged if you believe it is complete)."
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
            return self._finish(final, "final", ledger, trajectory, turns,
                                evidence=_signals(final=final, status="final"))
        except Exception as e:
            trajectory.event("error", error=f"{type(e).__name__}: {e}")
            log(_LOG, logging.ERROR, "query.error", query_id=query_id,
                error=f"{type(e).__name__}: {e}"[:300])
            return self._finish(None, "error", ledger, trajectory, turns,
                                evidence=_signals(final=None, status="error"))
        finally:
            await sandbox.close()
            trajectory.close()

    async def run_batch(self, questions: list[tuple[str, str]], env: EnvSpec, *,
                        run_dir: str | Path | None = None,
                        query_id: str | None = None) -> BatchQueryResult:
        """Answer several related questions in ONE loop over the corpus.

        questions: (qid, question_text) pairs. Exploration is shared —
        the root model answers all of them from one REPL session and
        submits via FINAL_BATCH. Budgets scale sub-linearly with the
        batch size (see scale_budgets). Questions the loop failed to
        answer come back as None in BatchQueryResult.answers; callers
        decide whether to retry those solo.
        """
        qids = [qid for qid, _ in questions]
        runner = replace(self, settings=scale_budgets(self.settings,
                                                      len(questions)))
        result = await runner.run(render_batch_task(questions), env,
                                  run_dir=run_dir, query_id=query_id,
                                  batch_questions=questions)
        parsed = coerce_batch(result.answer) or {}
        answers: dict[str, str | None] = {}
        evidence: dict[str, AnswerEvidence] = {}
        parent = result.evidence
        for qid in qids:
            value = parsed.get(qid)
            text = "" if value is None else str(value).strip()
            answers[qid] = text or None
            evidence[qid] = from_final(
                result.final, status=result.status, qid=qid,
                pushbacks=parent.pushbacks if parent else 0,
                negative_audit=parent.negative_audit if parent else "none",
                budget_warned=parent.budget_warned if parent else False,
                health_grade=parent.health_grade if parent else None,
                agreement=None,
            )
        return BatchQueryResult(answers=answers, result=result, evidence=evidence)

    async def run_batch_consensus(
        self, questions: list[tuple[str, str]], env: EnvSpec, *,
        run_dir: str | Path | None = None, query_id: str | None = None,
        passes: int = 2, tiebreak: bool = True,
    ) -> ConsensusBatchResult:
        """Answer a batch several times independently and vote per field.

        Repeated runs agree on ~99% of fields; the residual disagreements are
        where a run went wrong, and they are visible without a golden set
        because two independent passes rarely make the SAME mistake. Passes
        run concurrently (wall-clock unchanged, cost multiplied by `passes`)
        with different seeds, so the passes are as independent as the
        provider allows.

        Fields where the passes split are re-asked in their own focused loop
        and the vote is settled by that answer; a field that still cannot be
        settled comes back marked 'unresolved' rather than silently picking a
        side.
        """
        passes = max(1, passes)
        qids = [qid for qid, _ in questions]
        base_id = query_id or uuid.uuid4().hex[:12]

        async def one_pass(n: int) -> BatchQueryResult:
            # a distinct seed per pass: identical seeds invite identical
            # mistakes, which is exactly what voting cannot detect
            runner = replace(self, settings=replace(self.settings,
                                                    llm_seed=self.settings.llm_seed + n))
            return await runner.run_batch(questions, env, run_dir=run_dir,
                                          query_id=f"{base_id}_p{n}")

        pass_results = list(await asyncio.gather(
            *(one_pass(n) for n in range(passes))))

        answers: dict[str, ConsensusAnswer] = {}
        for qid in qids:
            votes = [pr.answers.get(qid) for pr in pass_results]
            tally: dict[str | None, int] = {}
            for vote in votes:
                key = _vote_key(vote)
                if key is not None:
                    tally[key] = tally.get(key, 0) + 1
            best_key, best_n = None, 0
            for key, n in tally.items():
                if n > best_n:
                    best_key, best_n = key, n
            chosen = next((v for v in votes if _vote_key(v) == best_key), None)
            agreement = best_n / passes
            if best_n == passes:
                resolved = "unanimous"
            elif best_n * 2 > passes:
                resolved = "majority"
            else:
                resolved = "split"
            winner = next((pr for pr in pass_results
                           if _vote_key(pr.answers.get(qid)) == best_key), None)
            ev = (winner.evidence.get(qid) if winner else None)
            if ev is not None:
                ev = from_final(
                    {"value": chosen, "is_var": False,
                     "verification": (winner.result.final or {}).get("verification")},
                    status=winner.result.status, qid=qid,
                    pushbacks=ev.pushbacks, negative_audit=ev.negative_audit,
                    budget_warned=ev.budget_warned, health_grade=ev.health_grade,
                    agreement=agreement, resolved_by=resolved, votes=votes,
                )
            answers[qid] = ConsensusAnswer(value=chosen, resolved_by=resolved,
                                           agreement=agreement, votes=votes,
                                           evidence=ev)

        contested = [qid for qid, a in answers.items() if a.contested]
        metrics().incr("consensus_fields", len(qids))
        metrics().incr("consensus_contested", len(contested))
        log(_LOG, logging.INFO, "consensus.voted", query_id=base_id,
            fields=len(qids), passes=passes, contested=contested)

        tiebreak_results: dict[str, QueryResult] = {}
        if contested and tiebreak:
            texts = dict(questions)

            async def settle(qid: str) -> tuple[str, QueryResult]:
                result = await self.run(texts[qid], env, run_dir=run_dir,
                                        query_id=f"{base_id}_tb_{qid}")
                return qid, result

            for qid, result in await asyncio.gather(
                    *(settle(qid) for qid in contested)):
                tiebreak_results[qid] = result
                text = "" if result.answer is None else str(result.answer).strip()
                prior = answers[qid]
                tb_ev = result.evidence
                if tb_ev is not None:
                    tb_ev = from_final(
                        result.final, status=result.status,
                        pushbacks=tb_ev.pushbacks,
                        negative_audit=tb_ev.negative_audit,
                        budget_warned=tb_ev.budget_warned,
                        health_grade=tb_ev.health_grade,
                        agreement=prior.agreement,
                        resolved_by="tiebreak" if text else "unresolved",
                        votes=[*prior.votes, text or None],
                    )
                answers[qid] = ConsensusAnswer(
                    value=text or prior.value,
                    resolved_by="tiebreak" if text else "unresolved",
                    agreement=prior.agreement,
                    votes=[*prior.votes, text or None],
                    evidence=tb_ev or prior.evidence)
            metrics().incr("consensus_tiebreaks", len(contested))

        return ConsensusBatchResult(
            answers=answers,
            pass_results=[pr.result for pr in pass_results],
            tiebreak_results=tiebreak_results)

    @staticmethod
    def _batch_gap(final: dict, qids: list[str]) -> str | None:
        """Structural completeness for FINAL_BATCH: every qid answered.

        No sub-LM involved — a missing id is objectively a gap. Returns the
        gap description, or None to accept.
        """
        answers = coerce_batch(final.get("value"))
        if answers is None:
            return ("the submitted value is not a dict of question id -> "
                    "answer. Use FINAL_BATCH({...}) with every question id "
                    "as a key.")
        missing = [q for q in qids if not str(answers.get(q, "") or "").strip()]
        if missing:
            return (f"no answer for question id(s): {', '.join(missing)}. "
                    'Every id must be present (use "NOT_FOUND" only when '
                    "the corpus truly lacks the answer).")
        return None

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
            if timeout <= 0 or ledger.spend_usd >= ledger.max_spend_usd:
                return None
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
            ledger.reserve_sub_call()
            async with asyncio.timeout(ledger.remaining_wall_s()):
                resp = await self.sub_client.complete(prompt, model=self.sub_model,
                                                      max_tokens=100)
            ledger.add_usage(resp.usage)
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
                breached: str | None = None,
                evidence: AnswerEvidence | None = None) -> QueryResult:
        snapshot = ledger.snapshot()
        trajectory.event("end", status=status, **snapshot)
        # .jsonl or .jsonl.enc, depending on trajectory encryption
        query_id = Path(trajectory.path).name.split(".", 1)[0]
        level = logging.INFO if status in ("final", "recovered") else logging.WARNING
        log(_LOG, level, "query.end", query_id=query_id, status=status,
            breached=breached, **snapshot)
        m = metrics()
        m.incr("queries_finished", status=status)
        m.incr("spend_usd", snapshot["spend_usd"])
        m.observe("query_latency_s", snapshot["wall_s"])
        m.observe("query_spend_usd", snapshot["spend_usd"])
        m.observe("root_iters", snapshot["root_iters"])
        if evidence is None:
            evidence = from_final(final, status=status)
        return QueryResult(
            answer=final.get("value") if final else None,
            status=status,
            final=final,
            ledger=ledger.snapshot(),
            trajectory_path=str(trajectory.path),
            iterations=ledger.root_iters,
            breached_cap=breached,
            evidence=evidence,
        )


def _normalize(text: str) -> str:
    out = re.sub(r"\s+", " ", text.strip().lower())
    return out[:300]
