"""Sanitized identities for evaluation runs and safe checkpoint reuse.

Document/question/gold contents are hashed, never copied into this manifest.
Provider keys and secret settings are neither read from the environment nor
serialized. A checkpoint must have the same complete identity before the
harness touches its results or starts a provider call.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import fields
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rnsr import __version__
from rnsr.eval.metrics import SCORER_VERSION

MANIFEST_VERSION = 1


class EvalResumeError(ValueError):
    """An evaluation directory cannot safely reuse its existing results."""


def _digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), default=str).encode()).hexdigest()


def _source_identity(source: Path) -> dict:
    source = Path(source)
    identity = {"path_sha256": _digest(str(source.resolve()))}
    try:
        digest = sha256()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        identity["content_sha256"] = digest.hexdigest()
    except OSError as exc:
        # Setup failures remain per-item results. If a file later becomes
        # readable, its changed identity requires a new run directory.
        identity["unavailable"] = type(exc).__name__
    return identity


def _settings_identity(settings) -> dict:
    values = {}
    for entry in fields(settings):
        value = getattr(settings, entry.name)
        secret = (not entry.repr or entry.name.lower().endswith(
            ("_secret", "_password", "_token", "_key")))
        if secret:
            values[entry.name] = {"configured": bool(value)}
        elif isinstance(value, Path):
            values[entry.name] = {"path_sha256": _digest(str(value.resolve()))}
        else:
            values[entry.name] = value
    return values


def _code_identity() -> dict:
    package = Path(__file__).resolve().parents[1]
    digest = sha256()
    # Includes loop prompts, provider adapters, ingestion and form policies,
    # even for an uncommitted checkout or installation without a Git repo.
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".json", ".txt", ".md"}:
            digest.update(str(path.relative_to(package)).encode() + b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    commit = None
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package.parent,
                                capture_output=True, text=True, timeout=2, check=False)
        candidate = result.stdout.strip()
        if result.returncode == 0 and len(candidate) == 40 \
                and all(c in "0123456789abcdef" for c in candidate):
            commit = candidate
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"package_version": __version__, "git_commit": commit,
            "package_content_sha256": digest.hexdigest()}


def _runtime_identity() -> dict:
    dependencies = {}
    for name in ("anthropic", "openai", "google-genai", "httpx", "pydantic", "numpy",
                 "sqlite-vec", "datasets", "docling", "pdfplumber", "camelot-py",
                 "pypdfium2", "firecrawl-anydoc", "extract-msg"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return {"python_version": list(sys.version_info[:3]), "dependencies": dependencies}


def build_run_manifest(items, system: str, runner, *, limit: int | None,
                       judge: bool, concurrency: int) -> dict:
    seen = set()
    dataset = []
    source_cache = {}
    for item in items:
        if item.qid in seen:
            raise ValueError("Evaluation qids must be unique; duplicate qid in dataset.")
        seen.add(item.qid)
        sources = []
        for source in item.sources:
            key = str(Path(source).resolve())
            if key not in source_cache:
                source_cache[key] = _source_identity(source)
            sources.append(source_cache[key])
        dataset.append({"qid_sha256": _digest(item.qid), "sources": sources,
                        "item_sha256": _digest({
                            "qid": item.qid, "question": item.question, "gold": item.gold,
                            "task_class": item.task_class, "context": item.context,
                            "expect": item.expect, "meta": item.meta, "sources": sources,
                        })})
    models = {}
    for role in ("root", "sub", "embed"):
        client = getattr(runner, f"{role}_client", None)
        models[role] = {"provider": getattr(client, "provider", None),
                        "model": getattr(runner, f"{role}_model", None)}
    identity = {
        "manifest_version": MANIFEST_VERSION,
        "scorer_version": SCORER_VERSION,
        "system": system,
        "models": models,
        "settings": _settings_identity(runner.settings),
        "code": _code_identity(),
        "runtime": _runtime_identity(),
        "dataset": {"n": len(items), "sha256": _digest(dataset), "items": dataset},
        "options": {"limit": limit, "judge": judge, "concurrency": concurrency},
    }
    return {"schema_version": MANIFEST_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "fingerprint": _digest(identity), "identity": identity}


def ensure_run_manifest(run_dir: Path, manifest: dict) -> dict:
    """Validate without writes on refusal; atomically create a new identity."""
    run_dir = Path(run_dir)
    path = run_dir / "run_manifest.json"
    guidance = (
        "Existing files were preserved. Use a fresh --run-dir for the new evaluation. "
        "Audit stored predictions without provider calls using "
        "python -m rnsr.eval.rescore --results <old-run>/results.jsonl --out <audit-dir>."
    )
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            intact = (stored["schema_version"] == MANIFEST_VERSION
                      and stored["fingerprint"] == _digest(stored["identity"]))
        except (OSError, ValueError, KeyError, TypeError):
            raise EvalResumeError(f"Cannot validate evaluation run manifest. {guidance}") from None
        if not intact or stored["fingerprint"] != manifest["fingerprint"]:
            raise EvalResumeError(
                "Evaluation identity changed (dataset, source content, settings, models, "
                f"system, code or scoring). {guidance}")
        return stored
    if any((run_dir / name).exists() for name in ("results.jsonl", "summary.json")):
        raise EvalResumeError(f"Legacy evaluation has no run manifest. {guidance}")
    run_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".run_manifest-", dir=run_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return manifest
