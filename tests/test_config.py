"""Settings: spec defaults and env overrides."""

from rnsr.config import Settings


def test_spec_defaults():
    s = Settings()
    # §7 budgets
    assert s.max_root_iters == 20
    assert s.max_sub_calls == 300
    assert s.max_wall_s == 600.0
    assert s.max_spend_usd == 2.0
    assert s.sub_concurrency == 16
    # §3.3 validation
    assert s.table_confidence_threshold == 0.7
    assert s.arithmetic_rel_tol == 0.005
    assert s.arithmetic_abs_tol == 1.0
    # §3.2 / §3.4 / §4.1 / §5
    assert s.coerce_threshold == 0.95
    assert (s.chunk_chars, s.chunk_overlap) == (1500, 200)
    assert s.sub_call_char_budget == 200_000
    assert s.expansion_max_rounds == 3


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no stray .env pickup
    monkeypatch.setenv("RNSR_MAX_ROOT_ITERS", "5")
    monkeypatch.setenv("RNSR_MAX_SPEND_USD", "0.25")
    monkeypatch.setenv("RNSR_ROOT_MODEL", "claude-sonnet-4-6")
    s = Settings.from_env()
    assert s.max_root_iters == 5
    assert s.max_spend_usd == 0.25
    assert s.root_model == "claude-sonnet-4-6"


def test_legacy_provider_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RNSR_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert Settings.from_env().provider == "gemini"
