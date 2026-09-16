"""Filesystem and process containment for the sandbox child (spec §4).

The OS launcher enforces containment. This audit hook provides clearer errors
and a second layer of restrictions; it is not a hostile-Python boundary.

Policy:
  - reads are confined to the interpreter's own installation (so
    ``import statistics`` still works mid-cell), the rnsr package, and
    the corpus artifact;
  - writes are confined to the private temp directory; corpus and sidecars
    are read-only, with annotation requests handled by the parent;
  - process creation and ctypes are refused outright — a child process
    is not audited at all, and ctypes reaches libc ``open()`` from below
    the audit layer, so both walk straight around every rule above.

Denials surface as PermissionError inside the cell, which the loop shows
the model as an ordinary observation: the message names the tools that
DO reach corpus text, because a model reaching for open() usually wants
a document it can get at legitimately.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from collections.abc import Iterable
from urllib.parse import unquote, urlsplit

# Events with no legitimate use in the child, where allowing the call at
# all would void the rest of the policy.
_BLOCKED_EVENTS = (
    # process creation is unnecessary for corpus analysis
    "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.startfile",
    "subprocess.Popen", "os.fork", "os.forkpty", "pty.spawn",
    # native calls bypass Python auditing (the OS boundary still applies)
    "ctypes.dlopen", "ctypes.dlsym", "ctypes.call_function", "ctypes.cdata",
    # the child holds RPC stubs for model/embedding calls; it needs no sockets
    "socket.",   # __new__, connect, bind, DNS, sendto, …
    "socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg",
    "socket.socket", "socket.getaddrinfo", "socket.gethostbyname",
    "socket.gethostbyname_ex",
)

# Path-bearing events that mutate the filesystem. Every string argument is
# checked, so two-path calls (rename, link) are covered without per-event
# argument indexes.
_WRITE_EVENTS = frozenset({
    "os.remove", "os.unlink", "os.rename", "os.replace", "os.rmdir",
    "os.mkdir", "os.makedirs", "os.chmod", "os.chown", "os.truncate",
    "os.link", "os.symlink", "os.utime", "os.setxattr", "os.removexattr",
    "shutil.copyfile", "shutil.copymode", "shutil.copystat",
    "shutil.copytree", "shutil.move", "shutil.rmtree",
    "shutil.unpack_archive", "shutil.make_archive",
})

# Path-bearing events that only read.
_READ_EVENTS = frozenset({"os.listdir", "os.scandir", "os.chdir", "glob.glob"})

_WRITE_MODE_CHARS = frozenset("wax+")
_WRITE_FLAGS = (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND
                | os.O_TRUNC | getattr(os, "O_EXCL", 0))

_TOOL_HINT = ("Corpus text is reachable without the filesystem: use `doc` "
              "(doc_id -> full text), `db` (SQL over extracted tables), or "
              "`search(query)`.")


def _norm(path: str) -> str:
    """Absolute, symlink-resolved path, for prefix comparison.

    realpath rather than abspath: '..' segments and symlinks are the
    obvious ways to walk out of an allowlisted directory. os.stat raises
    no audit event, so this cannot recurse into the hook.
    """
    try:
        return os.path.realpath(path)
    except (OSError, ValueError):
        return path


def default_read_dirs() -> list[str]:
    """Interpreter installation plus the rnsr package.

    Deliberately NOT the repo root or the current working directory: a
    dev checkout keeps .env next to the package, and the whole point is
    that provider keys stay unreachable.
    """
    import rnsr

    dirs = [sys.prefix, sys.base_prefix, os.path.dirname(os.__file__)]
    dirs += [p for p in sys.path if p and os.path.isdir(p)
             and _norm(p).startswith((_norm(sys.prefix), _norm(sys.base_prefix)))]
    dirs.append(os.path.dirname(os.path.abspath(rnsr.__file__)))
    return [_norm(d) for d in dirs if d]


def _under(path: str, roots: Iterable[str]) -> bool:
    return any(path == root or path.startswith(root + os.sep) for root in roots)


def install(*, corpus_db: str | None = None,
            read_dirs: Iterable[str] | None = None,
            extra_write_dirs: Iterable[str] = ()) -> None:
    """Install the audit hook. Irreversible for the life of the process."""
    read_roots = tuple(read_dirs if read_dirs is not None else default_read_dirs())
    write_roots = (_norm(tempfile.gettempdir()),
                   *(_norm(d) for d in extra_write_dirs))
    read_roots = read_roots + write_roots
    artifact = _norm(corpus_db) if corpus_db else ""
    busy = threading.local()
    # Capture policy helpers; module rebinding must not relax the hook.
    under, norm = _under, _norm
    blocked_events = _BLOCKED_EVENTS
    write_events, read_events = _WRITE_EVENTS, _READ_EVENTS
    write_modes, write_flags = _WRITE_MODE_CHARS, _WRITE_FLAGS
    tool_hint = _TOOL_HINT

    def _artifact_role(resolved: str) -> str | None:
        if not artifact:
            return None
        if resolved == artifact:
            return "exact"
        for suffix in ("-wal", "-shm", "-journal"):
            if resolved == artifact + suffix:
                return "sidecar"
        if resolved.startswith(artifact + "-mj"):
            return "sidecar"
        return None

    def hook(event: str, args) -> None:
        if event.startswith(blocked_events):
            if event.startswith("socket."):
                raise PermissionError(
                    f"network access is blocked in the sandbox ({event})")
            raise PermissionError(
                f"{event} is blocked in the sandbox: the child may not create "
                f"processes or load native libraries. {tool_hint}")
        if getattr(busy, "active", False):
            return
        mode = flags = None
        if event == "open":
            path, a1, a2 = (list(args) + [None, None, None])[:3]
            # Both builtin open and os.open emit (path, mode_or_None, flags).
            if isinstance(a1, str):
                mode, flags = a1, a2
            else:
                mode, flags = None, a2
            writing = bool(
                (isinstance(mode, str) and write_modes & set(mode))
                or (isinstance(flags, int) and flags & write_flags))
            paths, kind = [path], ("write" if writing else "read")
        elif event == "sqlite3.connect":
            raw = args[0] if args else None
            if isinstance(raw, str) and raw.startswith("file:"):
                raw = unquote(urlsplit(raw).path)
            if raw in (":memory:", "", None):
                return
            paths, kind = [raw], "read"
        elif event in write_events:
            paths, kind = list(args), "write"
        elif event in read_events:
            paths, kind = list(args), "read"
        else:
            return

        busy.active = True
        try:
            for raw in paths:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "replace")
                if not isinstance(raw, (str, os.PathLike)):
                    continue        # fd-based reopen: the open() already passed
                resolved = norm(os.fspath(raw))
                role = _artifact_role(resolved)
                if role and kind == "write":
                    raise PermissionError(
                        f"raw write to the corpus artifact is blocked. {tool_hint}")
                if role:
                    continue
                allowed = write_roots if kind == "write" else read_roots
                if under(resolved, allowed):
                    continue
                raise PermissionError(
                    f"filesystem {kind} of {resolved!r} is blocked in the "
                    f"sandbox. {tool_hint}")
        finally:
            busy.active = False

    sys.addaudithook(hook)

    # os.walk swallows scandir errors by default; force them through so a
    # listing of a blocked directory cannot succeed as an empty walk.
    _orig_walk = os.walk

    def _walk(top, topdown=True, onerror=None, followlinks=False):
        def _raise(err):
            raise err
        return _orig_walk(top, topdown=topdown, onerror=_raise,
                          followlinks=followlinks)

    os.walk = _walk  # type: ignore[method-assign]
