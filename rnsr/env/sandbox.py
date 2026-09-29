"""Parent-side sandbox: spawn the child REPL, exec cells, broker RPCs (§4).

The child never touches the network; tool stubs there RPC up to this
process, which owns provider traffic and the §7 concurrency semaphore via
the registered handlers. Wall-clock is enforced here with SIGKILL — a hung
cell kills the child, and the loop sees a SandboxError.

The child is spawned with a scrubbed environment (see _child_env): it
inherits only what the interpreter needs to start, so provider keys are
absent from the process that runs model-written code even before the
rnsr.env.fsguard audit hook denies it the filesystem.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import sqlite3
import struct
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rnsr.errors import PermanentProviderError, SandboxError

if TYPE_CHECKING:
    from rnsr.db.artifact import CorpusDB
    from rnsr.env.calculations import CalculationRegistry
    from rnsr.env.verify import Verifier

# op payload -> response body; e.g. {"op": "llm_batch", ...} -> {"results": [...]}
RpcHandler = Callable[[dict], Awaitable[dict]]

# Names the child interpreter needs to start and locate its packages.
# Everything else — API keys above all — is dropped: the child brokers
# every provider call through the parent and has no use for credentials.
_ENV_PASSTHROUGH = (
    "PATH", "VIRTUAL_ENV",
    "LANG", "LC_ALL", "LC_CTYPE",
    "SYSTEMROOT", "WINDIR", "PATHEXT", "COMSPEC",
)


def _child_env(scratch: str | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in _ENV_PASSTHROUGH}
    import rnsr

    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(rnsr.__file__))
    env["OPENBLAS_NUM_THREADS"] = env["OMP_NUM_THREADS"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"   # no ~/.local packages on the import path
    if scratch:
        # The guard allows writes under the child's own temp dir, so that
        # dir must be exclusively the child's: pointing TMPDIR at a private
        # scratch keeps SQLite's journals working without also handing the
        # cell read access to every other process's temp files.
        env["TMPDIR"] = env["TEMP"] = env["TMP"] = env["HOME"] = scratch
    return env


def _document_generation(conn, check=None) -> tuple:
    rows = []
    for row in conn.execute(
            "SELECT doc_id,sha256,content_sha256,ingested_at FROM documents ORDER BY doc_id"):
        if check is not None:
            check()
        rows.append(tuple(row))
    if check is not None:
        check()
    return tuple(rows)


@dataclass
class CellResult:
    ok: bool
    stdout: str = ""
    error: str | None = None
    final: dict | None = None       # {"value", "encoding", "is_var"} on FINAL/FINAL_VAR
    rpc_count: int = 0


@dataclass
class SandboxedRepl:
    """One persistent sandboxed Python session (namespace survives cells)."""

    rpc_handlers: dict[str, RpcHandler] = field(default_factory=dict)
    cpu_s: int = 300
    mem_bytes: int = 4 << 30
    fs_guard: bool = True
    _proc: asyncio.subprocess.Process | None = None
    _scratch: str | None = None
    _corpus: CorpusDB | None = None
    _verifier: Verifier | None = None
    _calculations: CalculationRegistry | None = None
    _init_options: dict = field(default_factory=dict)
    _source_generation: tuple = ()
    _source_identity: tuple = ()

    async def start(self, *, mode: str, context: str | None = None,
                    corpus_db: str | None = None, init_extra: dict | None = None) -> None:
        if self._proc is not None:
            if self._proc.returncode is None:
                raise SandboxError("sandbox session is already running")
            await self.close()
            self._proc = None
        try:
            self._init_options = dict(init_extra or {})
            if mode != "docdb" and self._init_options.get("metric_contract") is not None:
                raise ValueError("metric contracts require a docdb session")
            if mode == "docdb":
                from pathlib import Path

                from rnsr.db.artifact import CorpusDB
                from rnsr.db.schema import validate_frozen
                from rnsr.env.calculations import CalculationRegistry
                from rnsr.env.lazydoc import LazyDoc
                from rnsr.env.verify import Verifier

                corpus_db = str(Path(corpus_db).resolve())
                self._corpus = CorpusDB(corpus_db, mode="ro")
                validate_frozen(self._corpus.conn)
                self._verifier = Verifier(LazyDoc(self._corpus.conn))
                self._source_generation = _document_generation(self._corpus.conn)
                stat = self._corpus.path.stat()
                self._source_identity = (stat.st_dev, stat.st_ino)
                self._calculations = CalculationRegistry(
                    self._corpus.conn, metric_contract=self._init_options.get("metric_contract"))
            if self._scratch is None:
                self._scratch = os.path.realpath(tempfile.mkdtemp(prefix="rnsr-sandbox-"))
            from rnsr.env.osguard import launch_command

            self._proc = await asyncio.create_subprocess_exec(
                *launch_command(self._scratch, corpus_db),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=_child_env(self._scratch),
                start_new_session=True,
            )
            init = {"op": "init", "mode": mode, "context": context, "corpus_db": corpus_db,
                    "cpu_s": self.cpu_s, "mem_bytes": self.mem_bytes,
                    "fs_guard": self.fs_guard, **(init_extra or {})}
            result = await self._roundtrip(init, timeout=60.0)
            if not result.get("ok"):
                raise SandboxError(f"sandbox init failed: {result.get('error')}")
        except BaseException:
            await self.close()
            raise

    # --- protocol ----------------------------------------------------------

    def _send(self, msg: dict) -> None:
        assert self._proc and self._proc.stdin
        data = json.dumps(msg).encode()
        self._proc.stdin.write(struct.pack(">I", len(data)) + data)

    async def _recv(self) -> dict:
        assert self._proc and self._proc.stdout
        head = await self._proc.stdout.readexactly(4)
        (n,) = struct.unpack(">I", head)
        if n > 16 * 1024 * 1024:
            await self.kill()
            raise SandboxError("sandbox protocol frame exceeds 16 MiB")
        reply = json.loads((await self._proc.stdout.readexactly(n)).decode())
        if not isinstance(reply, dict):
            await self.kill()
            raise SandboxError("sandbox protocol frame must be an object")
        return reply

    async def _roundtrip(self, msg: dict, timeout: float, *, deadline: float | None = None,
                         pause_provider_rpc_timeout: bool = False) -> dict:
        """Serve a cell, bounding local work separately from provider waits.

        Uncapped queries may suspend the cell watchdog while a trusted
        provider/annotation handler runs. Those handlers must bound each
        provider request and propagate cancellation. Calculations and all
        other RPCs continue to share the local cell deadline.
        """
        deadline = time.monotonic() + timeout if deadline is None else deadline
        rpc_count = 0
        try:
            async with asyncio.timeout(max(0, deadline - time.monotonic())) as watchdog:
                if time.monotonic() >= deadline:
                    raise TimeoutError
                self._send(msg)
                while True:
                    reply = await self._recv()
                    if time.monotonic() >= deadline:
                        raise TimeoutError
                    if reply.get("kind") == "rpc":
                        rpc_count += 1
                        if (pause_provider_rpc_timeout
                                and reply.get("op") in {"llm_batch", "annotate", "classify"}):
                            remaining_local_s = deadline - time.monotonic()
                            if remaining_local_s <= 0:
                                raise TimeoutError
                            watchdog.reschedule(None)
                            await self._serve_rpc(reply)
                            # Provider wait does not consume local execution
                            # time, but a cheap/empty RPC must not replenish it.
                            deadline = time.monotonic() + remaining_local_s
                            watchdog.reschedule(
                                asyncio.get_running_loop().time() + remaining_local_s)
                        else:
                            await self._serve_rpc(reply, deadline=deadline)
                        continue
                    reply["_rpc_count"] = rpc_count
                    # Always replace any child-supplied value. Parent final
                    # verification shares the actual remaining local window.
                    reply["_deadline"] = deadline
                    return reply
        except (asyncio.CancelledError, PermanentProviderError):
            await self.kill()
            raise
        except TimeoutError:
            await self.kill()
            raise SandboxError(
                f"cell exceeded wall-clock limit ({timeout}s); sandbox killed"
            ) from None
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            await self.kill()
            raise SandboxError("sandbox sent an invalid protocol frame") from exc
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError) as e:
            await self.kill()
            raise SandboxError(f"sandbox died: {type(e).__name__}") from e

    async def _serve_rpc(self, request: dict, *, deadline: float | None = None) -> None:
        handler = (self._annotate if request.get("op") in {"annotate", "classify", "classification_counts"}
                   else self._classification if request.get("op") == "classification_final"
                   else self._calculation if request.get("op") == "calculation"
                   else self.rpc_handlers.get(request.get("op", "")))
        if handler is None:
            self._send({"error": f"no handler for rpc op {request.get('op')!r}"})
            return
        try:
            if request.get('op') in {'calculation', 'classification_final'}:
                body = await handler(request, deadline=deadline)
            else:
                body = await handler(request)
            self._send({"error": None, **body})
        except PermanentProviderError:
            # Authentication/configuration failures cannot be repaired by
            # model-written code; preserve them for the query's terminal error.
            raise
        except Exception as e:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("RPC exceeded cell wall-clock deadline") from e
            self._send({"error": f"{type(e).__name__}: {e}"})

    async def _annotate(self, request: dict) -> dict:
        """Trusted narrow writer; the child never receives a writable handle."""
        if self._corpus is None:
            raise ValueError("annotation requires a docdb session")
        import concurrent.futures
        import threading

        from rnsr.db.artifact import CorpusDB
        from rnsr.env.annotate import Annotator

        loop = asyncio.get_running_loop()
        cancelled = threading.Event()
        corpus_path = self._corpus.path

        def rpc(payload):
            handler = self.rpc_handlers.get(payload.get("op"))
            if handler is None:
                raise ValueError("annotation requires a configured llm_batch handler")
            if cancelled.is_set():
                raise RuntimeError("annotation cancelled")
            future = asyncio.run_coroutine_threadsafe(handler(payload), loop)
            while not cancelled.is_set():
                try:
                    return future.result(timeout=0.1)
                except concurrent.futures.TimeoutError:
                    pass
            future.cancel()
            raise RuntimeError("annotation cancelled")

        def apply():
            with CorpusDB(corpus_path, mode="rw") as corpus:
                corpus.conn.set_progress_handler(lambda: int(cancelled.is_set()), 1000)
                annotator = Annotator(
                    corpus.conn, rpc,
                    char_budget=self._init_options.get("sub_call_char_budget", 200_000),
                    default_batch_size=self._init_options.get("annotate_batch_size", 40),
                    model_identities=self._init_options.get('model_identities'),
                    cancelled=cancelled.is_set)
                if request.get('op') == 'classification_counts':
                    result = annotator.classification_counts(request.get('table'), request.get('column'))
                else:
                    classify = request.get('op') == 'classify'
                    method = annotator.classify if classify else annotator.annotate
                    keys = ("table", "new_col", "prompt", "where", "batch_size",
                            "model", "force", "votes", "allowed_labels")
                    if classify:
                        keys += ('expected_count',)
                    result = method(**{key: request[key] for key in keys if key in request})
                return {"result": result}

        try:
            return await asyncio.to_thread(apply)
        except BaseException:
            cancelled.set()
            raise

    def _check_source(self) -> None:
        stat = self._corpus.path.stat()
        if ((stat.st_dev, stat.st_ino) != self._source_identity or
                _document_generation(self._corpus.conn, self._calculations._check_deadline)
                != self._source_generation):
            raise ValueError("source changed; restart the sandbox session")

    async def _classification(self, request: dict, *, deadline: float | None = None) -> dict:
        if self._corpus is None:
            raise ValueError('classification aggregates require a docdb session')
        from rnsr.env.classification import classification_final

        if request.keys() - {'kind', 'op', 'table', 'column', 'operation', 'labels'}:
            raise ValueError('unsupported classification arguments; counts cannot be supplied')
        try:
            with self._calculations.deadline(deadline):
                self._corpus.conn.execute('BEGIN')
                self._check_source()
                answer, report = classification_final(self._corpus.conn, **{
                    key: request[key] for key in ('table', 'column', 'operation', 'labels') if key in request},
                    cancelled=lambda: deadline is not None and time.monotonic() >= deadline)
                self._calculations._check_deadline()
                return {'result': {'answer': answer, 'verification': report}}
        finally:
            self._corpus.conn.rollback()

    async def _calculation(self, request: dict, *, deadline: float | None = None) -> dict:
        if self._corpus is None or self._calculations is None:
            raise ValueError("source calculations require a docdb session")
        action = request.get('action')
        actions = {
            'source': (self._calculations.source_number,
                       {'table', 'rowid', 'column', 'unit_span', 'period_span'}),
            'source_text': (self._calculations.source_text_number,
                            {'doc_id', 'char_start', 'char_end', 'unit_span', 'period_span'}),
            'financial': (self._calculations.calculate_financial, {'metric', 'inputs', 'convention'}),
            'render': (self._calculations.resolve_finals, {'template', 'results', 'decimals'}),
            'compute': (self._calculations.calculate, {'operation', 'operand_ids'}),
            'get': (self._calculations.get, {'record_id'}),
            'metric': (self._calculations.calculate_metric, set()),
        }
        if not isinstance(action, str) or action not in actions:
            raise ValueError("unknown calculation action")
        fn, allowed = actions[action]
        arguments = {k: v for k, v in request.items() if k not in {'kind', 'op', 'action'}}
        if arguments.keys() - allowed:
            raise ValueError("unsupported calculation arguments; values cannot be supplied")
        try:
            with self._calculations.deadline(deadline):
                self._corpus.conn.execute('BEGIN')
                self._check_source()
                return {'result': fn(**arguments)}
        finally:
            # The progress handler is reset before rollback, even on timeout.
            self._corpus.conn.rollback()

    # --- public API ----------------------------------------------------------

    async def exec_cell(self, code: str, *, timeout: float = 120.0,
                        pause_provider_rpc_timeout: bool = False) -> CellResult:
        deadline = time.monotonic() + timeout
        reply = await self._roundtrip(
            {"op": "exec", "code": code}, timeout, deadline=deadline,
            pause_provider_rpc_timeout=pause_provider_rpc_timeout)
        deadline = reply.get("_deadline", deadline)
        final = reply.get("final")
        if final is not None and self._verifier is not None:
            from rnsr.env.finalize import submitted_quotes, validate_final

            self._verifier.set_deadline(deadline)
            try:
                # Keep source generation and all reads on one SQLite snapshot;
                # trusted concurrent ingest must not invalidate cached evidence.
                with self._calculations.deadline(deadline):
                    self._corpus.conn.execute("BEGIN")
                    self._check_source()
                    value = final["value"]
                    report = final.get('verification')
                    if isinstance(report, dict) and report.get('check') == 'source_bound_calculation':
                        value, report = self._calculations.resolve_final(report.get('calculation_id'), value)
                        final['value'], final['verification'] = value, report
                    elif isinstance(report, dict) and report.get('check') == 'source_bound_calculations':
                        actual, proof = self._calculations.resolve_finals(
                            report.get('template'), report.get('results'), decimals=report.get('decimals'))
                        if value != actual:
                            raise ValueError('answer differs from parent-rendered calculation values')
                        final['verification'] = proof
                    elif isinstance(report, dict) and report.get('check') == 'classification_aggregate':
                        from rnsr.env.classification import classification_final

                        self._calculations.validate_legacy_final(value)
                        actual, proof = classification_final(self._corpus.conn,
                            report.get('table'), report.get('column'), report.get('operation'),
                            report.get('labels'), cancelled=lambda: time.monotonic() >= deadline)
                        if report.get('annotation_version') != proof['classification']['annotation_version']:
                            raise ValueError('classification annotation changed after the draft was computed')
                        if value != actual:
                            raise ValueError('answer differs from parent classification aggregate')
                        final['verification'] = proof
                    else:
                        batch = isinstance(value, dict)
                        abstention = self._calculations.validate_legacy_final(value)
                        final["verification"] = abstention if abstention is not None else validate_final(
                            value, submitted_quotes(report, batch=batch), self._verifier, batch=batch)
            except (TimeoutError, sqlite3.OperationalError) as exc:
                if isinstance(exc, TimeoutError) or time.monotonic() >= deadline:
                    await self.kill()
                    raise SandboxError("source verification exceeded cell wall-clock deadline") from exc
                raise
            except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
                return CellResult(ok=False, stdout=reply.get("stdout", ""),
                                  error=f"Parent final verification rejected: {exc}",
                                  rpc_count=reply.get("_rpc_count", 0))
            finally:
                self._corpus.conn.rollback()
                self._verifier.set_deadline(None)
        return CellResult(
            ok=reply.get("ok", False),
            stdout=reply.get("stdout", ""),
            error=reply.get("error"),
            final=final,
            rpc_count=reply.get("_rpc_count", 0),
        )

    async def vars(self) -> dict[str, dict]:
        reply = await self._roundtrip({"op": "vars"}, timeout=30.0)
        return reply.get("vars", {})

    async def kill(self) -> None:
        if self._proc and self._proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._proc.pid, signal.SIGKILL)
            await self._proc.wait()

    async def close(self) -> None:
        self._calculations = None
        if self._proc and self._proc.returncode is None:
            try:
                self._send({"op": "shutdown"})
                async with asyncio.timeout(5):
                    await self._proc.wait()
            except Exception:
                await self.kill()
        if self._verifier is not None:
            self._verifier.close()
            self._verifier = None
        if self._corpus is not None:
            self._corpus.close()
            self._corpus = None
        if self._scratch:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None

    async def __aenter__(self) -> SandboxedRepl:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
