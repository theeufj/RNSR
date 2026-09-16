"""Direct escape tests for the sandbox filesystem/process guard."""

import pytest

from rnsr.env.sandbox import SandboxedRepl
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.pipeline import ingest


def _parse(path):
    return ParsedDocument(
        doc_id="acme", source_path=str(path), sha256="d" * 64, n_pages=1,
        parser="fake",
        elements=[Element("text", "hello world", 1)],
        tables=[RawTable(
            page=1, header=["Item", "Amt"],
            rows=[["A", "1"], ["Total", "1"]], extractor="fake",
        )],
    )


@pytest.fixture
def corpus(tmp_path):
    out = tmp_path / "corpus.db"
    ingest([tmp_path / "a.pdf"], out, parse=_parse)
    return out


@pytest.fixture
async def repl(corpus):
    r = SandboxedRepl()
    await r.start(mode="docdb", corpus_db=str(corpus))
    yield r
    await r.close()


class TestEscapeSuite:
    async def test_home_and_env_unreadable(self, repl):
        res = await repl.exec_cell(
            "import os\n"
            "try:\n"
            "    open(os.path.expanduser('~/.ssh/id_rsa')).read()\n"
            "    print('LEAK')\n"
            "except (PermissionError, FileNotFoundError):\n"
            "    print('blocked')\n"
        )
        assert "blocked" in res.stdout
        assert "LEAK" not in res.stdout

    async def test_cwd_dotenv_unreadable(self, repl):
        res = await repl.exec_cell(
            "try:\n"
            "    open('.env').read()\n"
            "    print('LEAK')\n"
            "except (PermissionError, FileNotFoundError):\n"
            "    print('blocked')\n"
        )
        assert "blocked" in res.stdout
        assert "LEAK" not in res.stdout

    async def test_dotdot_escape(self, repl):
        res = await repl.exec_cell(
            "try:\n"
            "    open('../../../../etc/passwd').read()\n"
            "    print('LEAK')\n"
            "except PermissionError:\n"
            "    print('blocked')\n"
        )
        assert "blocked" in res.stdout

    async def test_os_open_write_flags(self, repl, corpus):
        res = await repl.exec_cell(
            "import os\n"
            f"p = {str(corpus)!r}\n"
            "try:\n"
            "    os.open(p, os.O_TRUNC | os.O_WRONLY)\n"
            "    print('OPENED')\n"
            "except PermissionError:\n"
            "    print('blocked')\n"
        )
        assert "blocked" in res.stdout
        assert "OPENED" not in res.stdout

    async def test_subprocess_popen_fork_and_popen(self, repl):
        res = await repl.exec_cell(
            "import os, subprocess\n"
            "hits = []\n"
            "for name, fn in (\n"
            "    ('system', lambda: os.system('echo x')),\n"
            "    ('popen', lambda: os.popen('echo x').read()),\n"
            "    ('Popen', lambda: subprocess.Popen(['echo', 'x'])),\n"
            "):\n"
            "    try:\n"
            "        fn(); hits.append(name)\n"
            "    except PermissionError:\n"
            "        hits.append(name + '-blocked')\n"
            "try:\n"
            "    os.fork(); hits.append('fork')\n"
            "except PermissionError:\n"
            "    hits.append('fork-blocked')\n"
            "print(hits)\n"
        )
        for name in ("system", "popen", "Popen", "fork"):
            assert f"{name}-blocked" in res.stdout
            assert f"'{name}'" not in res.stdout.replace(f"{name}-blocked", "")

    async def test_multiprocessing_blocked(self, repl):
        res = await repl.exec_cell(
            "import multiprocessing as mp, os\n"
            "try:\n"
            "    mp.get_context('fork').Process(target=os._exit, args=(0,)).start()\n"
            "    print('SPAWNED')\n"
            "except PermissionError:\n"
            "    print('blocked')\n"
        )
        assert "blocked" in res.stdout
        assert "SPAWNED" not in res.stdout

    async def test_listdir_scandir_glob_walk_outside(self, repl):
        res = await repl.exec_cell(
            "import os, glob\n"
            "hits = []\n"
            "for name, fn in (\n"
            "    ('listdir', lambda: os.listdir('/etc')),\n"
            "    ('scandir', lambda: list(os.scandir('/etc'))),\n"
            "    ('glob', lambda: glob.glob('/etc/*')),\n"
            "    ('walk', lambda: next(os.walk('/etc'), None)),\n"
            "):\n"
            "    try:\n"
            "        fn(); hits.append(name)\n"
            "    except PermissionError:\n"
            "        hits.append(name + '-blocked')\n"
            "print(hits)\n"
        )
        for name in ("listdir", "scandir", "glob", "walk"):
            assert f"{name}-blocked" in res.stdout

    async def test_shutil_and_rename_two_path(self, repl):
        res = await repl.exec_cell(
            "import os, shutil, tempfile\n"
            "src = os.path.join(tempfile.gettempdir(), 'rnsr_src.txt')\n"
            "open(src, 'w').write('x')\n"
            "hits = []\n"
            "try:\n"
            "    shutil.copy(src, '/etc/rnsr_escape')\n"
            "    hits.append('copy')\n"
            "except PermissionError:\n"
            "    hits.append('copy-blocked')\n"
            "try:\n"
            "    os.rename(src, '/etc/rnsr_escape')\n"
            "    hits.append('rename')\n"
            "except PermissionError:\n"
            "    hits.append('rename-blocked')\n"
            "print(hits)\n"
        )
        assert "copy-blocked" in res.stdout
        assert "rename-blocked" in res.stdout

    async def test_sqlite_connect_outside_and_attach(self, repl):
        res = await repl.exec_cell(
            "import sqlite3\n"
            "hits = []\n"
            "try:\n"
            "    sqlite3.connect('/etc/rnsr_escape.db')\n"
            "    hits.append('connect')\n"
            "except PermissionError:\n"
            "    hits.append('connect-blocked')\n"
            "try:\n"
            "    db.execute(\"ATTACH DATABASE '/etc/rnsr_escape.db' AS x\")\n"
            "    hits.append('attach')\n"
            "except Exception:\n"
            "    hits.append('attach-blocked')\n"
            "print(hits)\n"
        )
        assert "connect-blocked" in res.stdout
        assert "attach-blocked" in res.stdout

    async def test_raw_write_to_artifact_denied(self, repl, corpus):
        res = await repl.exec_cell(
            f"p = {str(corpus)!r}\n"
            "try:\n"
            "    open(p, 'wb').write(b'nope')\n"
            "    print('WROTE')\n"
            "except PermissionError:\n"
            "    print('blocked')\n"
        )
        assert "blocked" in res.stdout
        assert "WROTE" not in res.stdout
        assert corpus.exists() and corpus.stat().st_size > 0

    async def test_wal_sidecar_write_denied(self, repl, corpus):
        res = await repl.exec_cell(
            f"p = {str(corpus)!r} + '-wal'\n"
            "open(p, 'wb').write(b'ok')\n"
            "print(open(p, 'rb').read())\n"
        )
        assert not res.ok
        assert "raw write" in res.error

    async def test_dns_and_socket_blocked(self, repl):
        res = await repl.exec_cell(
            "import socket\n"
            "hits = []\n"
            "try:\n"
            "    socket.socket(); hits.append('sock')\n"
            "except PermissionError:\n"
            "    hits.append('sock-blocked')\n"
            "try:\n"
            "    socket.getaddrinfo('example.com', 80); hits.append('dns')\n"
            "except PermissionError:\n"
            "    hits.append('dns-blocked')\n"
            "try:\n"
            "    socket.gethostbyname('example.com'); hits.append('host')\n"
            "except PermissionError:\n"
            "    hits.append('host-blocked')\n"
            "print(hits)\n"
        )
        assert "sock-blocked" in res.stdout
        assert "dns-blocked" in res.stdout
        assert "host-blocked" in res.stdout

    async def test_dunder_stdout_is_not_the_protocol_pipe(self, repl):
        res = await repl.exec_cell(
            "import sys\n"
            "sys.__stdout__.write('INJECT')\n"
            "print('ok')\n"
        )
        assert res.ok, res.error
        assert res.stdout.strip() == "ok"
        follow = await repl.exec_cell("print('still-alive')")
        assert follow.ok, follow.error
        assert follow.stdout.strip() == "still-alive"

    async def test_final_spoof_by_name_is_not_accepted(self, repl):
        res = await repl.exec_cell(
            "class _FinalAnswer(Exception):\n"
            "    def __init__(self, value, is_var=False, verification=None):\n"
            "        self.value = value; self.is_var = is_var\n"
            "        self.verification = verification\n"
            "raise _FinalAnswer('spoofed', False)\n"
        )
        assert res.final is None
        assert not res.ok

    async def test_real_final_still_short_circuits(self, repl):
        res = await repl.exec_cell("FINAL('from-tools', quotes=['hello world'])")
        assert res.ok, res.error
        assert res.final and res.final["value"] == "from-tools"

    async def test_importlib_outside_path(self, repl):
        res = await repl.exec_cell(
            "import importlib.machinery\n"
            "try:\n"
            "    importlib.machinery.SourceFileLoader('x', '/etc/passwd')"
            ".get_data('/etc/passwd')\n"
            "    print('loaded')\n"
            "except PermissionError:\n"
            "    print('blocked')\n"
        )
        assert "blocked" in res.stdout
        assert "loaded" not in res.stdout

    async def test_rlimit_is_installed(self, repl):
        # Linux CI often refuses RLIMIT_CPU (stays unlimited / -1) while still
        # accepting RLIMIT_AS. Either finite cap means the installer ran.
        res = await repl.exec_cell(
            "import resource\n"
            "def finite(soft):\n"
            "    return soft not in (-1, resource.RLIM_INFINITY) and soft > 0\n"
            "cpu = resource.getrlimit(resource.RLIMIT_CPU)[0]\n"
            "mem = resource.getrlimit(resource.RLIMIT_AS)[0]\n"
            "print(finite(cpu) or finite(mem))\n"
        )
        assert res.ok, res.error
        if res.stdout.strip() != "True":
            pytest.skip("kernel refused both RLIMIT_CPU and RLIMIT_AS")
        assert res.stdout.strip() == "True"


class TestHostilePythonRegressions:
    async def test_guard_global_rebinding_does_not_read_sentinel(self, repl, tmp_path):
        sentinel = tmp_path / 'private-sentinel'
        sentinel.write_text('PRIVATE-SENTINEL')
        result = await repl.exec_cell(
            "import rnsr.env.fsguard as fg\n"
            "fg._under = lambda *_: True\nfg._BLOCKED_EVENTS = ()\n"
            f"print(open({str(sentinel)!r}).read())")
        assert not result.ok
        assert 'PRIVATE-SENTINEL' not in result.stdout

    @pytest.mark.parametrize('operation', [
        'os.remove(p)', 'os.truncate(p, 0)',
        'os.rename(p, p + ".moved")',
        'os.open(p, os.O_RDWR | os.O_TRUNC)',
        'open(p + "-journal", "wb").write(b"corrupt")',
        'open(p + "-shm", "wb").write(b"corrupt")',
    ])
    async def test_artifact_and_sidecars_are_read_only(self, repl, corpus, operation):
        before = corpus.read_bytes()
        result = await repl.exec_cell(f'import os\np = {str(corpus)!r}\n{operation}')
        assert not result.ok
        assert corpus.read_bytes() == before

    async def test_sql_readonly_survives_authorizer_removal(self, repl, corpus):
        result = await repl.exec_cell(
            "db.set_authorizer(None)\n"
            "db.execute('DROP TABLE doc_text')")
        assert not result.ok and 'readonly' in result.error.lower()
        result = await repl.exec_cell("print(doc['acme'])")
        assert result.stdout.strip() == 'hello world'

    async def test_forged_child_verification_rejected_by_parent(self, repl):
        result = await repl.exec_cell(
            "from rnsr.env.final_answer import FinalAnswer\n"
            "raise FinalAnswer('invented', is_var=False, verification={"
            "'passed': True, 'quotes': [{'quote': 'forged retained text', "
            "'matched': True, 'doc_id': 'acme', 'char_start': 0, 'char_end': 10}]})")
        assert not result.ok and result.final is None
        assert 'Parent final verification rejected' in result.error

    async def test_threads_cannot_bypass_path_guard(self, repl, tmp_path):
        sentinel = tmp_path / 'thread-sentinel'
        sentinel.write_text('THREAD-SECRET')
        result = await repl.exec_cell(
            'import threading, tempfile, os\n'
            'leaks = []\n'
            'def allowed():\n'
            '    for _ in range(300):\n'
            '        with open(os.path.join(tempfile.gettempdir(), "probe"), "w") as f: f.write("x")\n'
            'def denied():\n'
            '    for _ in range(300):\n'
            '        try:\n'
            f'            leaks.append(open({str(sentinel)!r}).read())\n'
            '        except PermissionError: pass\n'
            'threads = [threading.Thread(target=fn) for fn in [allowed, denied] * 3]\n'
            '[t.start() for t in threads]\n[t.join() for t in threads]\n'
            'print(leaks)')
        assert result.ok, result.error
        assert result.stdout.strip() == '[]'

    async def test_kernel_blocks_native_read_without_python_audit(self, tmp_path):
        sentinel = tmp_path / 'native-sentinel'
        sentinel.write_text('NATIVE-SECRET')
        async with SandboxedRepl(fs_guard=False) as repl:
            await repl.start(mode='classic', context='')
            result = await repl.exec_cell(
                'import ctypes, os\n'
                'libc = ctypes.CDLL(None, use_errno=True)\n'
                f'fd = libc.open({str(sentinel).encode()!r}, os.O_RDONLY)\n'
                'print(fd)\n'
                'if fd >= 0: print(os.read(fd, 100))')
            assert result.ok, result.error
            assert result.stdout.strip() == '-1'

    async def test_parent_annotation_rejects_source_column(self, repl):
        result = await repl.exec_cell("semantic_annotate('t_acme_001', 'item', 'rewrite')")
        assert not result.ok
        assert 'cannot overwrite a source' in result.error


@pytest.mark.parametrize('operation', [
    'os.open(p, os.O_RDWR | os.O_TRUNC)',
    'os.remove(p)',
    'open(p + "-wal", "wb").write(b"corrupt")',
])
async def test_kernel_protects_source_with_audit_disabled(corpus, operation):
    before = corpus.read_bytes()
    async with SandboxedRepl(fs_guard=False) as repl:
        await repl.start(mode='docdb', corpus_db=str(corpus))
        result = await repl.exec_cell(f'import os\np = {str(corpus)!r}\n{operation}')
        assert not result.ok
    assert corpus.read_bytes() == before


async def test_kernel_isolates_host_network_with_audit_disabled():
    import socket

    # Use a test-owned endpoint; never probe a real service or external host.
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        port = listener.getsockname()[1]
        async with SandboxedRepl(fs_guard=False) as repl:
            await repl.start(mode='classic', context='')
            result = await repl.exec_cell(
                'import socket\n'
                f'socket.create_connection(("127.0.0.1", {port}), timeout=0.2)\n'
                'print("CONNECTED")')
            assert not result.ok
            assert 'CONNECTED' not in result.stdout
