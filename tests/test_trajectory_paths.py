"""Query IDs are log metadata, never filesystem paths."""

import csv
import json
import os
import unicodedata

import pytest

from rnsr.eval.audit import export_audit
from rnsr.harness.trajectory import TrajectoryWriter, read_trajectory, trajectory_stem


@pytest.mark.parametrize("query_id", ["q1", "query-123", "finance_bench.0001"])
def test_safe_ids_keep_existing_filenames(tmp_path, query_id):
    with TrajectoryWriter(tmp_path, query_id) as writer:
        writer.event("start")
    assert writer.path == tmp_path / f"{query_id}.jsonl"
    assert read_trajectory(writer.path)[0]["query_id"] == query_id


@pytest.mark.parametrize("query_id", [
    "a/b", "a\\b", "../escape", "../../escape", "/tmp/absolute-query",
    ".", "..", "", "q\x00id", "中文/問題", "café", "a" * 10000,
    "CON", "con", "aux.txt", "q1.", "Q1", "_qid_literal",
])
def test_untrusted_ids_stay_inside_run_root(tmp_path, query_id):
    run_dir = tmp_path / "run"
    with TrajectoryWriter(run_dir, query_id) as writer:
        writer.event("start", query_id="cannot override original")
    assert writer.path.parent == run_dir
    assert len(writer.path.name.encode()) < 150
    assert writer.path.name == trajectory_stem(query_id) + ".jsonl"
    assert read_trajectory(writer.path)[0]["query_id"] == query_id
    assert list(tmp_path.iterdir()) == [run_dir]


def test_absolute_id_cannot_write_to_named_path(tmp_path):
    outside = tmp_path / "outside"
    with TrajectoryWriter(tmp_path / "logs", str(outside)) as writer:
        writer.event("start")
    assert not outside.exists() and not outside.with_suffix(".jsonl").exists()


def test_distinct_ids_cannot_collide_through_sanitizing_or_filesystem_folding(tmp_path):
    ids = ["a/b", "a_b", "a\\b", "q1", "Q1", "café",
           unicodedata.normalize("NFD", "café"), "a" * 300 + "x", "a" * 300 + "y"]
    ids.append(trajectory_stem("a/b"))  # generated basenames have a reserved namespace
    names = [trajectory_stem(qid) for qid in ids]
    assert len({name.casefold() for name in names}) == len(ids)
    for qid in ids:
        with TrajectoryWriter(tmp_path, qid) as writer:
            writer.event("start")
    assert len(list(tmp_path.iterdir())) == len(ids)


def test_same_id_reopens_same_file_and_appends(tmp_path):
    for kind in ("start", "end"):
        with TrajectoryWriter(tmp_path, "path/query") as writer:
            writer.event(kind)
    assert [r["kind"] for r in read_trajectory(writer.path)] == ["start", "end"]


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_existing_alias_cannot_modify_outside_file(tmp_path, link_kind):
    outside = tmp_path / "sentinel.jsonl"
    outside.write_text("unchanged\n")
    run_dir = tmp_path / "logs"
    run_dir.mkdir()
    alias = run_dir / "q1.jsonl"
    if link_kind == "symlink":
        alias.symlink_to(outside)
    else:
        os.link(outside, alias)
    with pytest.raises((OSError, ValueError)), TrajectoryWriter(run_dir, "q1") as writer:
        writer.event("start")
    assert outside.read_text() == "unchanged\n"


def test_nonregular_log_does_not_block_on_open(tmp_path):
    os.mkfifo(tmp_path / "q1.jsonl")
    with pytest.raises((OSError, ValueError)):
        TrajectoryWriter(tmp_path, "q1")


def test_encrypted_unsafe_id_retains_original(tmp_path):
    pytest.importorskip("cryptography")
    from cryptography.fernet import Fernet

    key = Fernet.generate_key().decode()
    with TrajectoryWriter(tmp_path, "../../secret", key=key) as writer:
        writer.event("start")
    assert writer.path.name.endswith(".jsonl.enc")
    assert "../../secret" not in writer.path.read_text()
    assert read_trajectory(writer.path, key)[0]["query_id"] == "../../secret"


@pytest.mark.parametrize("mode", ["full", "metadata", "redacted"])
def test_long_identifier_remains_joinable_in_each_content_mode(tmp_path, mode):
    qid = "long-id-" * 100
    with TrajectoryWriter(tmp_path, qid, content=mode) as writer:
        writer.event("start")
    assert read_trajectory(writer.path)[0]["query_id"] == qid


def test_audit_uses_original_id_for_answers_and_status_join(tmp_path):
    qid = "../../client/問題.1"
    work = tmp_path / "work"
    with TrajectoryWriter(work / "trajectories", qid) as writer:
        writer.event("start", question="Question?")
        writer.event("final", value={qid: "42"})
        writer.event("end", status="final")
    with (work / "answers_status.csv").open("w") as fh:
        csv_writer = csv.DictWriter(fh, fieldnames=["query_id", "status", "tier"])
        csv_writer.writeheader()
        csv_writer.writerow({"query_id": qid, "status": "final", "tier": "high"})
    out = tmp_path / "packet"
    export_audit(work, out)
    evidence_path = out / "evidence" / f"{trajectory_stem(qid)}.json"
    evidence = json.loads(evidence_path.read_text())
    assert evidence["qid"] == qid and evidence["answer"] == "42"
    assert evidence["tier"] == "high"
    with (out / "review.csv").open() as fh:
        assert list(csv.DictReader(fh))[0]["qid"] == qid
    assert sorted(p.name for p in tmp_path.iterdir()) == ["packet", "work"]


@pytest.mark.parametrize("encrypted", [False, True])
def test_audit_preserves_dotted_legacy_filename(tmp_path, encrypted):
    qid = "Legacy.ID.001"
    work = tmp_path / "work"
    work.mkdir()
    lines = [json.dumps({"kind": "start", "question": "Question?"}),
             json.dumps({"kind": "final", "value": "42"})]
    key, suffix = "", ".jsonl"
    if encrypted:
        pytest.importorskip("cryptography")
        from cryptography.fernet import Fernet

        key = Fernet.generate_key().decode()
        cipher = Fernet(key.encode())
        lines = [cipher.encrypt(line.encode()).decode() for line in lines]
        suffix += ".enc"
    (work / f"{qid}{suffix}").write_text("\n".join(lines))
    out = tmp_path / "packet"
    export_audit(work, out, key=key)
    evidence = json.loads((out / "evidence" / f"{trajectory_stem(qid)}.json").read_text())
    assert evidence["qid"] == qid and evidence["answer"] == "42"


def test_audit_evidence_replaces_alias_without_modifying_outside_target(tmp_path):
    work, out = tmp_path / "work", tmp_path / "out"
    with TrajectoryWriter(work / "trajectories", "q1") as writer:
        writer.event("start")
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text("unchanged")
    evidence_dir = out / "evidence"
    evidence_dir.mkdir(parents=True)
    alias = evidence_dir / "q1.json"
    alias.symlink_to(sentinel)
    export_audit(work, out)
    assert sentinel.read_text() == "unchanged"
    assert not alias.is_symlink()
    assert json.loads(alias.read_text())["qid"] == "q1"
