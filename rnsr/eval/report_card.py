"""Corpus report card written next to answer-csv output."""

from __future__ import annotations

import json
from pathlib import Path


def write_report_card(
    out_dir: str | Path,
    *,
    report: dict | None = None,
    health: dict | None = None,
    ledger: dict | None = None,
) -> Path:
    """Write report.md: health, tier distribution, autopsy, cost."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = report or {}
    if health is None:
        health = report.get("health") or {}
    if ledger is None:
        for name in ("autopsy.json", "loss-ledger.json"):
            p = out / name
            if p.exists():
                ledger = json.loads(p.read_text())
                break
        ledger = ledger or {}

    tiers = report.get("tier_counts") or {}
    spend = ((report.get("provider") or {}).get("spend_usd")
             or (report.get("metrics") or {}).get("spend_usd") or 0)
    lines = [
        "# Corpus report card",
        "",
        f"- questions: {report.get('questions', report.get('answers_written', ''))}",
        f"- health: {health.get('grade', 'unknown')}",
        f"- spend_usd: {spend}",
        f"- wall_s: {report.get('wall_s', '')}",
        "",
        "## Trust tiers",
        "",
        f"- high: {tiers.get('high', 0)}",
        f"- medium: {tiers.get('medium', 0)}",
        f"- low: {tiers.get('low', 0)}",
        "",
    ]
    if health.get("findings"):
        lines += ["## Health findings", ""]
        for f in health["findings"]:
            if isinstance(f, dict):
                lines.append(f"- [{f.get('severity')}] {f.get('detail')}")
            else:
                lines.append(f"- {f}")
        lines.append("")
    if ledger.get("cause_counts"):
        lines += ["## Autopsy (misses by cause)", ""]
        for cause, n in ledger["cause_counts"].items():
            lines.append(f"- {cause}: {n}")
        lines.append("")
        if ledger.get("accuracy") is not None:
            lines.append(f"accuracy: {ledger['accuracy']:.1%}")
            lines.append("")
    path = out / "report.md"
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path
