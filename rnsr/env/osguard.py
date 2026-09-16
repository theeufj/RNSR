"""OS-enforced boundary for generated Python; audit hooks are supplementary.

The child gets read-only runtime/source mounts and one private scratch directory.
There is deliberately no uncontained fallback when the platform policy fails.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
from pathlib import Path

from rnsr.env.fsguard import default_read_dirs
from rnsr.errors import SandboxError


def launch_command(scratch: str, corpus_db: str | None) -> list[str]:
    python = [sys.executable, "-m", "rnsr.env.sandbox_child"]
    roots = sorted(set(default_read_dirs()))
    import rnsr

    files = [os.path.dirname(os.path.dirname(rnsr.__file__))]
    if corpus_db:
        path = os.path.realpath(corpus_db)
        files += [path, *(path + suffix for suffix in ("-wal", "-shm", "-journal"))]
    if sys.platform == "darwin":
        launcher = "/usr/bin/sandbox-exec"
        if not os.path.isfile(launcher):
            raise SandboxError("OS sandbox unavailable: macOS sandbox-exec is required")
        # Seatbelt restrictions inherit across exec and cannot be lifted by
        # mutating Python globals, invoking native code, or creating threads.
        roots.extend(["/System/Library", "/usr/lib", "/usr/share/locale"])
        # Framework/Homebrew Python extensions may link outside sys.prefix.
        # Grant exact already-loaded dependency files, never an entire user
        # installation directory (which can also contain credentials).
        import ctypes
        import sqlite3  # noqa: F401 - load the SQLite runtime dependency
        with contextlib.suppress(ImportError):
            import numpy  # noqa: F401 - preload its native runtime dependencies
        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        files.extend(os.path.realpath(dyld._dyld_get_image_name(i).decode())
                     for i in range(dyld._dyld_image_count()))
        read_rules = "\n".join(f"(subpath {json.dumps(p)})" for p in roots)
        read_rules += "\n" + "\n".join(f"(literal {json.dumps(p)})" for p in files)
        read_rules += '\n(literal "/dev/urandom") (literal "/dev/random")'
        profile = f'''(version 1)
(deny default)
(allow file-read-metadata)
(allow file-read-data (literal "/"))
(allow file-read* {read_rules} (subpath {json.dumps(scratch)}))
(allow file-write* (subpath {json.dumps(scratch)}) (literal "/dev/null"))
(allow file-read* (literal "/dev/null"))
(allow process-exec (subpath {json.dumps(os.path.realpath(sys.base_prefix))}))
(allow process-info* (target self))
(allow signal (target self))
(allow sysctl-read)
'''
        profile_path = Path(scratch) / "sandbox.sb"
        profile_path.write_text(profile)
        return [launcher, "-f", str(profile_path), *python]
    if sys.platform == "linux":
        launcher = shutil.which("bwrap")
        if not launcher:
            raise SandboxError("OS sandbox unavailable: install bubblewrap and enable user namespaces")
        # A private PID namespace hides host processes, a private network
        # namespace removes connectivity, and every visible runtime is ro.
        cmd = [launcher, "--die-with-parent", "--new-session", "--unshare-all",
               "--cap-drop", "ALL", "--dev", "/dev"]
        roots.extend(p for p in ("/usr", "/lib", "/lib64") if os.path.exists(p))
        # Avoid shadowing mounts when a virtualenv lives below a runtime root.
        roots = [p for p in sorted(set(roots)) if not any(
            p != q and p.startswith(q + "/") for q in roots)]
        for path in roots:
            cmd.extend(["--ro-bind", path, path])
        for path in ["/etc/ld.so.cache", *(p for p in files if not os.path.isdir(p))]:
            if os.path.exists(path):
                cmd.extend(["--ro-bind", path, path])
        cmd.extend(["--bind", scratch, scratch, "--chdir", scratch, "--", *python])
        return cmd
    raise SandboxError(f"OS sandbox unsupported on {sys.platform}; refusing generated code")
