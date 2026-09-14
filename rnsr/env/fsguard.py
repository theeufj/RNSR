"""Filesystem and process containment for the sandbox child (spec §4).

The audit hook installed here is what stands between model-written code
and the rest of the machine. Matter documents arrive from opposing
parties, so the threat is not an accident: it is a poisoned document that
talks the root model into ``open('~/.aws/credentials').read()`` or
``os.system(...)``. Sockets alone are not enough of a boundary.

Policy:
  - reads are confined to the interpreter's own installation (so
    ``import statistics`` still works mid-cell), the rnsr package, and
    the corpus artifact;
  - writes are confined to the corpus artifact (annotation columns) and
    the temp directory;
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
from collections.abc import Iterable

# Events with no legitimate use in the child, where allowing the call at
# all would void the rest of the policy.
_BLOCKED_EVENTS = (
    # an unaudited child process would sidestep every path rule below
    "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.startfile",
    "subprocess.Popen", "os.fork", "os.forkpty", "pty.spawn",
    # ctypes calls libc open() beneath the audit layer
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
    busy = False

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
        nonlocal busy
        if event.startswith(_BLOCKED_EVENTS):
            if event.startswith("socket."):
                raise PermissionError(
                    f"network access is blocked in the sandbox ({event})")
            raise PermissionError(
                f"{event} is blocked in the sandbox: the child may not create "
                f"processes or load native libraries. {_TOOL_HINT}")
        if busy:
            return
        mode = flags = None
        if event == "open":
            path, a1, a2 = (list(args) + [None, None, None])[:3]
            # builtin open: (path, mode_str, flags); os.open: (path, flags, perm)
            if isinstance(a1, str):
                mode, flags = a1, a2
            else:
                mode, flags = None, a1
            writing = bool(
                (isinstance(mode, str) and _WRITE_MODE_CHARS & set(mode))
                or (isinstance(flags, int) and flags & _WRITE_FLAGS))
            paths, kind = [path], ("write" if writing else "read")
        elif event == "sqlite3.connect":
            raw = args[0] if args else None
            if raw in (":memory:", "", None):
                return
            paths, kind = [raw], "read"
        elif event in _WRITE_EVENTS:
            paths, kind = list(args), "write"
        elif event in _READ_EVENTS:
            paths, kind = list(args), "read"
        else:
            return

        busy = True
        try:
            for raw in paths:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "replace")
                if not isinstance(raw, (str, os.PathLike)):
                    continue        # fd-based reopen: the open() already passed
                resolved = _norm(os.fspath(raw))
                role = _artifact_role(resolved)
                if role == "exact" and kind == "write" and isinstance(mode, str) and (
                        "w" in mode or "x" in mode):
                    raise PermissionError(
                        f"raw write to the corpus artifact is blocked. {_TOOL_HINT}")
                if role:
                    continue
                allowed = write_roots if kind == "write" else read_roots
                if _under(resolved, allowed):
                    continue
                raise PermissionError(
                    f"filesystem {kind} of {resolved!r} is blocked in the "
                    f"sandbox. {_TOOL_HINT}")
        finally:
            busy = False

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
