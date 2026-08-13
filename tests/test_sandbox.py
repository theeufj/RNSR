"""Sandbox: persistence, FINAL, RPC brokering, limits, network block."""

import pytest

from rnsr.env.sandbox import SandboxedRepl
from rnsr.errors import SandboxError


@pytest.fixture
async def repl():
    r = SandboxedRepl()
    await r.start(mode="classic", context="The needle is 42. " * 100)
    yield r
    await r.close()


class TestExec:
    async def test_stdout_captured(self, repl):
        res = await repl.exec_cell("print('hello', 1 + 1)")
        assert res.ok and res.stdout == "hello 2\n"

    async def test_namespace_persists_across_cells(self, repl):
        await repl.exec_cell("x = 41")
        res = await repl.exec_cell("print(x + 1)")
        assert res.stdout == "42\n"

    async def test_exception_reported_not_fatal(self, repl):
        res = await repl.exec_cell("1 / 0")
        assert not res.ok
        assert "ZeroDivisionError" in res.error
        follow = await repl.exec_cell("print('still alive')")
        assert follow.ok

    async def test_context_preloaded(self, repl):
        res = await repl.exec_cell("print(len(context), context[:13])")
        assert res.stdout.startswith("1800 The needle is")


class TestFinal:
    async def test_final_short_circuits(self, repl):
        res = await repl.exec_cell("FINAL('the answer')\nprint('unreachable')")
        assert res.final == {"value": "the answer", "encoding": "json",
                             "is_var": False, "verification": None}
        assert "unreachable" not in res.stdout

    async def test_final_var_json_value(self, repl):
        res = await repl.exec_cell("pairs = [(1, 2), (3, 4)]\nFINAL_VAR(pairs)")
        assert res.final["is_var"] is True
        assert res.final["value"] == [[1, 2], [3, 4]]

    async def test_final_var_repr_fallback(self, repl):
        res = await repl.exec_cell("FINAL_VAR({1, 2})")  # sets are not JSON
        assert res.final["encoding"] == "repr"
        assert res.final["value"] == "{1, 2}"


class TestRpc:
    async def test_llm_query_brokered(self):
        async def handler(req):
            return {"results": [f"echo:{p}" for p in req["prompts"]]}

        r = SandboxedRepl(rpc_handlers={"llm_batch": handler})
        await r.start(mode="classic", context="")
        try:
            res = await r.exec_cell("print(llm_query('what is x?'))")
            assert res.stdout == "echo:what is x?\n"
            assert res.rpc_count == 1
            res = await r.exec_cell("print(llm_map(['a', 'b']))")
            assert res.stdout == "['echo:a', 'echo:b']\n"
        finally:
            await r.close()

    async def test_rpc_error_surfaces_in_cell(self):
        async def handler(req):
            raise RuntimeError("provider down")

        r = SandboxedRepl(rpc_handlers={"llm_batch": handler})
        await r.start(mode="classic", context="")
        try:
            res = await r.exec_cell("llm_query('x')")
            assert not res.ok
            assert "provider down" in res.error
        finally:
            await r.close()

    async def test_missing_handler(self, repl):
        res = await repl.exec_cell("llm_query('x')")
        assert not res.ok
        assert "no handler" in res.error


class TestVars:
    async def test_vars_excludes_preloaded(self, repl):
        await repl.exec_cell("answer = 3234\n_hidden = 1")
        vars_ = await repl.vars()
        assert vars_["answer"] == {"type": "int", "repr": "3234"}
        assert "_hidden" not in vars_
        assert "context" not in vars_ and "FINAL" not in vars_

    async def test_long_reprs_truncated(self, repl):
        await repl.exec_cell("blob = 'x' * 10000")
        vars_ = await repl.vars()
        assert len(vars_["blob"]["repr"]) <= 501


class TestContainment:
    async def test_infinite_loop_killed(self, repl):
        with pytest.raises(SandboxError, match="wall-clock"):
            await repl.exec_cell("while True: pass", timeout=1.5)

    async def test_network_blocked(self, repl):
        res = await repl.exec_cell(
            "import socket\n"
            "s = socket.socket()\n"
            "s.connect(('127.0.0.1', 9))\n"
        )
        assert not res.ok
        assert "network access is blocked" in res.error

    async def test_dead_sandbox_raises(self, repl):
        with pytest.raises(SandboxError, match="sandbox died"):
            await repl.exec_cell("import os; os._exit(1)")


class TestFilesystemContainment:
    """Matter documents are untrusted input; a cell that reads them must not
    also be able to read the operator's secrets or spawn processes."""

    async def test_reading_outside_corpus_blocked(self, repl, tmp_path):
        secret = tmp_path / ".env"
        secret.write_text("ANTHROPIC_API_KEY=sk-should-never-be-read")
        res = await repl.exec_cell(f"print(open({str(secret)!r}).read())")
        assert not res.ok
        assert "filesystem read" in res.error
        assert "sk-should-never-be-read" not in res.stdout

    async def test_home_directory_traversal_blocked(self, repl):
        res = await repl.exec_cell(
            "from pathlib import Path\n"
            "print(Path('~/.ssh/id_rsa').expanduser().read_text())")
        assert not res.ok
        assert "filesystem read" in res.error

    async def test_writing_outside_corpus_blocked(self, repl, tmp_path):
        target = tmp_path / "escape.txt"
        res = await repl.exec_cell(f"open({str(target)!r}, 'w').write('x')")
        assert not res.ok
        assert "filesystem write" in res.error
        assert not target.exists()

    async def test_subprocess_blocked(self, repl):
        # a child process is not audited, so this is the widest escape
        res = await repl.exec_cell("import os; os.system('echo pwned')")
        assert not res.ok
        assert "blocked in the sandbox" in res.error

    async def test_ctypes_blocked(self, repl):
        res = await repl.exec_cell(
            "import ctypes; ctypes.CDLL(None)")
        assert not res.ok
        assert "blocked in the sandbox" in res.error

    async def test_stdlib_imports_still_work(self, repl):
        # the guard must not break ordinary analysis mid-cell
        res = await repl.exec_cell(
            "import statistics, itertools, collections\n"
            "print(statistics.median([3, 1, 2]))")
        assert res.ok, res.error
        assert res.stdout.strip() == "2"

    async def test_numpy_still_importable(self, repl):
        # numpy dlopens on import, which the guard forbids; the child
        # pre-warms it so a cell's `import numpy` needs no native loading
        res = await repl.exec_cell(
            "import numpy as np\nprint(int(np.array([1, 2, 3]).sum()))")
        assert res.ok, res.error
        assert res.stdout.strip() == "6"

    async def test_tempfile_still_writable(self, repl):
        res = await repl.exec_cell(
            "import tempfile, os\n"
            "p = os.path.join(tempfile.gettempdir(), 'rnsr_probe.txt')\n"
            "open(p, 'w').write('ok')\n"
            "print(open(p).read())\n"
            "os.remove(p)")
        assert res.ok, res.error
        assert res.stdout.strip() == "ok"

    async def test_guard_can_be_disabled_for_debugging(self, tmp_path):
        from rnsr.env.sandbox import SandboxedRepl

        probe = tmp_path / "readable.txt"
        probe.write_text("plain")
        r = SandboxedRepl(fs_guard=False)
        await r.start(mode="classic", context="")
        try:
            res = await r.exec_cell(f"print(open({str(probe)!r}).read())")
            assert res.ok, res.error
            assert res.stdout.strip() == "plain"
        finally:
            await r.close()


class TestEnvironmentScrubbing:
    async def test_provider_keys_absent_from_child(self, repl, monkeypatch):
        res = await repl.exec_cell(
            "import os\n"
            "print([k for k in os.environ if 'KEY' in k.upper()])")
        assert res.ok, res.error
        assert res.stdout.strip() == "[]"

    def test_child_env_drops_secrets_keeps_path(self, monkeypatch):
        from rnsr.env.sandbox import _child_env

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        monkeypatch.setenv("PATH", "/usr/bin")
        env = _child_env()
        assert env["PATH"] == "/usr/bin"
        assert "ANTHROPIC_API_KEY" not in env
        assert "OPENAI_API_KEY" not in env
