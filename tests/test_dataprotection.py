"""Trajectory redaction/encryption/retention and the work-dir lock."""

import json
import time

import pytest

from rnsr.harness.trajectory import (
    TrajectoryWriter,
    prune_trajectories,
    read_trajectory,
    redact,
)
from rnsr.runlock import WorkDirBusy, WorkDirLock

PRIVILEGED = ("Daniel Robert Mitchell resides at 17 Strathfield Avenue "
              "and his email is daniel.mitchell@email.com.au. " * 4)


class TestRedaction:
    def test_full_mode_keeps_everything(self, tmp_path):
        with TrajectoryWriter(tmp_path, "q1") as w:
            w.event("cell", code="print(doc)", stdout=PRIVILEGED, ok=True)
        text = (tmp_path / "q1.jsonl").read_text()
        assert "Strathfield" in text

    def test_redacted_mode_drops_content_but_keeps_shape(self, tmp_path):
        with TrajectoryWriter(tmp_path, "q1", content="redacted") as w:
            w.event("cell", code="print(doc)", stdout=PRIVILEGED, ok=True,
                    rpc_count=3)
        record = json.loads((tmp_path / "q1.jsonl").read_text().strip())
        assert "Strathfield" not in json.dumps(record)
        assert record["ok"] is True and record["rpc_count"] == 3
        assert record["stdout"].startswith("<redacted ")
        assert "sha256:" in record["stdout"]

    def test_digest_is_stable_so_runs_stay_comparable(self):
        a = redact({"stdout": PRIVILEGED}, "redacted")["stdout"]
        b = redact({"stdout": PRIVILEGED}, "redacted")["stdout"]
        c = redact({"stdout": PRIVILEGED + "!"}, "redacted")["stdout"]
        assert a == b and a != c

    def test_metadata_mode_omits_content_keys(self, tmp_path):
        with TrajectoryWriter(tmp_path, "q1", content="metadata") as w:
            w.event("cell", code="print(doc)", stdout=PRIVILEGED, ok=True)
        record = json.loads((tmp_path / "q1.jsonl").read_text().strip())
        assert "stdout" not in record and "code" not in record
        assert record["ok"] is True

    def test_operational_fields_survive_redaction(self):
        out = redact({"kind": "end", "status": "final", "spend_usd": 0.12,
                      "cap": "max_wall_s"}, "redacted")
        assert out == {"kind": "end", "status": "final", "spend_usd": 0.12,
                       "cap": "max_wall_s"}


class TestEncryption:
    @pytest.fixture
    def key(self):
        cryptography = pytest.importorskip("cryptography")  # noqa: F841
        from cryptography.fernet import Fernet

        return Fernet.generate_key().decode()

    def test_lines_are_unreadable_at_rest(self, tmp_path, key):
        with TrajectoryWriter(tmp_path, "q1", key=key) as w:
            w.event("cell", stdout=PRIVILEGED)
        path = tmp_path / "q1.jsonl.enc"
        assert path.exists()
        raw = path.read_text()
        assert "Strathfield" not in raw and "stdout" not in raw

    def test_round_trip_through_reader(self, tmp_path, key):
        with TrajectoryWriter(tmp_path, "q1", key=key) as w:
            w.event("start", question="what is the date?")
            w.event("end", status="final")
        records = read_trajectory(tmp_path / "q1.jsonl.enc", key)
        assert [r["kind"] for r in records] == ["start", "end"]
        assert records[0]["question"] == "what is the date?"

    def test_wrong_key_cannot_read(self, tmp_path, key):
        from cryptography.fernet import Fernet, InvalidToken

        with TrajectoryWriter(tmp_path, "q1", key=key) as w:
            w.event("start", question="secret")
        with pytest.raises(InvalidToken):
            read_trajectory(tmp_path / "q1.jsonl.enc",
                            Fernet.generate_key().decode())


class TestRetention:
    def test_prunes_only_expired_files(self, tmp_path):
        old, new = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
        old.write_text("{}\n")
        new.write_text("{}\n")
        stale = time.time() - 10 * 86400
        import os

        os.utime(old, (stale, stale))
        assert prune_trajectories(tmp_path, max_age_days=7) == 1
        assert not old.exists() and new.exists()

    def test_zero_disables_retention(self, tmp_path):
        (tmp_path / "a.jsonl").write_text("{}\n")
        assert prune_trajectories(tmp_path, max_age_days=0) == 0
        assert (tmp_path / "a.jsonl").exists()

    def test_encrypted_trajectories_are_pruned_too(self, tmp_path):
        import os

        path = tmp_path / "old.jsonl.enc"
        path.write_text("token\n")
        stale = time.time() - 5 * 86400
        os.utime(path, (stale, stale))
        assert prune_trajectories(tmp_path, max_age_days=1) == 1


class TestWorkDirLock:
    def test_second_holder_is_refused(self, tmp_path):
        first = WorkDirLock(tmp_path, label="run-a").acquire()
        try:
            with pytest.raises(WorkDirBusy, match="in use by another rnsr run"):
                WorkDirLock(tmp_path, label="run-b").acquire()
        finally:
            first.release()

    def test_lock_is_reusable_after_release(self, tmp_path):
        WorkDirLock(tmp_path).acquire().release()
        second = WorkDirLock(tmp_path).acquire()
        second.release()

    def test_holder_details_recorded(self, tmp_path):
        import os

        with WorkDirLock(tmp_path, label="answer-csv q.csv"):
            holder = json.loads((tmp_path / ".rnsr.lock").read_text())
        assert holder["pid"] == os.getpid()
        assert holder["label"] == "answer-csv q.csv"

    def test_context_manager_releases(self, tmp_path):
        with WorkDirLock(tmp_path):
            pass
        WorkDirLock(tmp_path).acquire().release()
