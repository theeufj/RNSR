"""Sandbox child process: persistent REPL namespace + RPC stubs (spec §4).

Run as ``python -m rnsr.env.sandbox_child``. Speaks length-prefixed JSON
over the original stdin/stdout; user code's print() goes to an in-memory
buffer, never the protocol channel.

Hardening (accident containment, not adversarial isolation):
  - resource rlimits on CPU time and address space
  - an audit hook that blocks socket use — the child needs zero network;
    llm_query()/embed() are RPC stubs served by the parent
  - the parent enforces per-cell wall-clock with SIGKILL

Ops: init (preload namespace), exec (run a cell), vars (namespace summary
for the variable-recovery fallback), shutdown.
"""

from __future__ import annotations

import contextlib
import io
import json
import struct
import sys
import traceback

_BLOCKED_AUDIT_PREFIXES = ("socket.connect", "socket.bind", "socket.sendto",
                           "socket.sendmsg")


class _FinalAnswer(Exception):
    def __init__(self, value, is_var: bool, verification: dict | None = None):
        self.value = value
        self.is_var = is_var
        self.verification = verification


def _install_limits(cpu_s: int, mem_bytes: int) -> None:
    import resource

    with contextlib.suppress(Exception):  # some rlimits unsupported per-OS
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
    with contextlib.suppress(Exception):
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    def hook(event: str, args) -> None:
        if event.startswith(_BLOCKED_AUDIT_PREFIXES):
            raise PermissionError(f"network access is blocked in the sandbox ({event})")

    sys.addaudithook(hook)


class Channel:
    """Length-prefixed JSON over binary streams."""

    def __init__(self, rfile, wfile):
        self._r = rfile
        self._w = wfile

    def send(self, msg: dict) -> None:
        data = json.dumps(msg, default=repr).encode()
        self._w.write(struct.pack(">I", len(data)) + data)
        self._w.flush()

    def recv(self) -> dict | None:
        head = self._r.read(4)
        if len(head) < 4:
            return None
        (n,) = struct.unpack(">I", head)
        return json.loads(self._r.read(n).decode())


def _jsonable(value):
    try:
        json.dumps(value)
        return value, "json"
    except (TypeError, ValueError):
        return repr(value), "repr"


class Child:
    def __init__(self, channel: Channel):
        self.channel = channel
        self.namespace: dict = {}
        self.preloaded: set[str] = set()

    # --- RPC up to the parent (used by tool stubs) -------------------------

    def rpc(self, payload: dict) -> dict:
        self.channel.send({"kind": "rpc", **payload})
        resp = self.channel.recv()
        if resp is None:
            raise RuntimeError("parent closed the channel mid-RPC")
        if resp.get("error"):
            raise RuntimeError(f"rpc failed: {resp['error']}")
        return resp

    def _llm_query(self, prompt: str, model: str = "sub") -> str:
        return self.rpc({"op": "llm_batch", "prompts": [str(prompt)],
                         "model": model})["results"][0]

    def _llm_map(self, prompts: list[str], model: str = "sub") -> list[str]:
        return self.rpc({"op": "llm_batch", "prompts": [str(p) for p in prompts],
                         "model": model})["results"]

    # --- ops ---------------------------------------------------------------

    def op_init(self, msg: dict) -> dict:
        _install_limits(msg.get("cpu_s", 300), msg.get("mem_bytes", 4 << 30))

        def FINAL(answer):  # noqa: N802 — spec-mandated name (§4)
            raise _FinalAnswer(answer, is_var=False)

        def FINAL_VAR(value):  # noqa: N802
            raise _FinalAnswer(value, is_var=True)

        self.namespace = {
            "llm_query": self._llm_query,
            "llm_map": self._llm_map,
            "FINAL": FINAL,
            "FINAL_VAR": FINAL_VAR,
        }
        if msg.get("mode") == "classic":
            self.namespace["context"] = msg.get("context", "")
        elif msg.get("mode") == "docdb":
            from rnsr.env.tools import build_namespace

            self.namespace.update(build_namespace(msg["corpus_db"], self, msg))
        self.preloaded = set(self.namespace)
        return {"ok": True, "mode": msg.get("mode")}

    def op_exec(self, msg: dict) -> dict:
        code = msg["code"]
        out = io.StringIO()
        final = None
        error = None
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(compile(code, "<cell>", "exec"), self.namespace)  # noqa: S102
        except BaseException as exc:
            # Matched by name: under `python -m` this module is __main__, so
            # tools.py's `from rnsr.env.sandbox_child import _FinalAnswer`
            # yields a distinct class object for the same code.
            if type(exc).__name__ == "_FinalAnswer":
                value, encoding = _jsonable(exc.value)
                final = {"value": value, "encoding": encoding,
                         "is_var": exc.is_var,
                         "verification": getattr(exc, "verification", None)}
            else:
                error = traceback.format_exc(limit=8)
        return {"ok": error is None, "stdout": out.getvalue(), "final": final,
                "error": error}

    def op_vars(self, msg: dict) -> dict:
        summaries = {}
        for name, value in self.namespace.items():
            if name in self.preloaded or name.startswith("_"):
                continue
            r = repr(value)
            summaries[name] = {"type": type(value).__name__,
                               "repr": r[:500] + ("…" if len(r) > 500 else "")}
        return {"ok": True, "vars": summaries}

    def serve(self) -> None:
        while True:
            msg = self.channel.recv()
            if msg is None or msg.get("op") == "shutdown":
                return
            handler = getattr(self, f"op_{msg.get('op', '')}", None)
            if handler is None:
                self.channel.send({"kind": "result", "ok": False,
                                   "error": f"unknown op {msg.get('op')!r}"})
                continue
            try:
                result = handler(msg)
            except BaseException:
                result = {"ok": False, "error": traceback.format_exc(limit=8)}
            self.channel.send({"kind": "result", **result})


def main() -> None:
    # Grab the real pipes before user code can touch sys.stdout.
    channel = Channel(sys.stdin.buffer, sys.stdout.buffer)
    sys.stdout = sys.stderr  # stray writes go to stderr, not the protocol
    Child(channel).serve()


if __name__ == "__main__":
    main()
