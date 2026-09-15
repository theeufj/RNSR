"""Playbook loading and financial addon selection."""

from rnsr.harness.playbook import Playbook, load_playbook
from rnsr.harness.prompts.base import render_system


def test_default_system_includes_financial_addon():
    text = render_system("docdb", manifest={})
    assert "quick ratio" in text


def test_empty_addons_omits_financial(tmp_path):
    path = tmp_path / "playbook.json"
    path.write_text('{"addons": []}')
    pb = load_playbook(path)
    assert pb is not None and pb.addons == []
    text = render_system("docdb", manifest={}, playbook=pb)
    assert "quick ratio" not in text


def test_playbook_prompt_block_lists_aliases():
    pb = Playbook(entity_aliases={"Acme": ["ACME Inc", "Acme Co"]},
                  addons=[])
    block = pb.prompt_block()
    assert "Acme" in block and "ACME Inc" in block
