"""Parent-side sandbox: spawn the child REPL, exec cells, broker RPCs (§4).

The child never touches the network; tool stubs there RPC up to this
process, which owns provider traffic and the §7 concurrency semaphore via
the registered handlers. Wall-clock is enforced here with SIGKILL — a hung
cell kills the child, and the loop sees a SandboxError.
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from rnsr.errors import SandboxError

# op payload -> response body; e.g. {"op": "llm_batch", ...} -> {"results": [...]}
RpcHandler = Callable[[dict], Awaitable[dict]]


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
    _proc: asyncio.subprocess.Process | None = None

    async def start(self, *, mode: str, context: str | None = None,
                    corpus_db: str | None = None, init_extra: dict | None = None) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "rnsr.env.sandbox_child",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        init = {"op": "init", "mode": mode, "context": context, "corpus_db": corpus_db,
                "cpu_s": self.cpu_s, "mem_bytes": self.mem_bytes, **(init_extra or {})}
        result = await self._roundtrip(init, timeout=60.0)
        if not result.get("ok"):
            raise SandboxError(f"sandbox init failed: {result.get('error')}")

    # --- protocol ----------------------------------------------------------

    def _send(self, msg: dict) -> None:
        assert self._proc and self._proc.stdin
        data = json.dumps(msg).encode()
        self._proc.stdin.write(struct.pack(">I", len(data)) + data)

    async def _recv(self) -> dict:
        assert self._proc and self._proc.stdout
        head = await self._proc.stdout.readexactly(4)
        (n,) = struct.unpack(">I", head)
        return json.loads((await self._proc.stdout.readexactly(n)).decode())

    async def _roundtrip(self, msg: dict, timeout: float) -> dict:
        """Send an op and read to its result, serving RPCs along the way."""
        self._send(msg)
        rpc_count = 0
        try:
            async with asyncio.timeout(timeout):
                while True:
                    reply = await self._recv()
                    if reply.get("kind") == "rpc":
                        rpc_count += 1
                        await self._serve_rpc(reply)
                        continue
                    reply["_rpc_count"] = rpc_count
                    return reply
        except TimeoutError:
            await self.kill()
            raise SandboxError(
                f"cell exceeded wall-clock limit ({timeout}s); sandbox killed"
            ) from None
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError) as e:
            await self.kill()
            raise SandboxError(f"sandbox died: {type(e).__name__}") from e

    async def _serve_rpc(self, request: dict) -> None:
        handler = self.rpc_handlers.get(request.get("op", ""))
        if handler is None:
            self._send({"error": f"no handler for rpc op {request.get('op')!r}"})
            return
        try:
            body = await handler(request)
            self._send({"error": None, **body})
        except Exception as e:
            self._send({"error": f"{type(e).__name__}: {e}"})

    # --- public API ----------------------------------------------------------

    async def exec_cell(self, code: str, *, timeout: float = 120.0) -> CellResult:
        reply = await self._roundtrip({"op": "exec", "code": code}, timeout)
        return CellResult(
            ok=reply.get("ok", False),
            stdout=reply.get("stdout", ""),
            error=reply.get("error"),
            final=reply.get("final"),
            rpc_count=reply.get("_rpc_count", 0),
        )

    async def vars(self) -> dict[str, dict]:
        reply = await self._roundtrip({"op": "vars"}, timeout=30.0)
        return reply.get("vars", {})

    async def kill(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.kill()
            await self._proc.wait()

    async def close(self) -> None:
        if self._proc and self._proc.returncode is None:
            try:
                self._send({"op": "shutdown"})
                async with asyncio.timeout(5):
                    await self._proc.wait()
            except Exception:
                await self.kill()

    async def __aenter__(self) -> SandboxedRepl:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
