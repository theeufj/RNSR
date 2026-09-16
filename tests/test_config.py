"""Settings: spec defaults and env overrides."""

import pytest

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
    assert s.health_min_validation_rate == 0.7
    assert s.health_max_untranscribed_pages == 0
    assert s.transcribe_scans == "auto"
    assert s.allow_degraded is False
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
    with pytest.warns(DeprecationWarning, match="RNSR_PROVIDER"):
        assert Settings.from_env().provider == "gemini"


@pytest.mark.parametrize("field,value", [
    ("chunk_chars", 0), ("chunk_overlap", -1), ("chunk_overlap", 1500),
    ("sub_concurrency", 0), ("max_wall_s", 0), ("cell_timeout_s", -1),
    ("max_spend_usd", float("nan")), ("max_wall_s", float("inf")),
    ("coerce_threshold", 1.1), ("transcribe_scans", "typo"),
    ("log_format", "typo"), ("provider", "typo"),
])
def test_invalid_settings_rejected(field, value):
    with pytest.raises(ValueError):
        Settings(**{field: value})


def test_optional_path_and_boolean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RNSR_SERVICE_CORPUS_ROOT", str(tmp_path))
    monkeypatch.setenv("RNSR_SANDBOX_FS_GUARD", "false")
    settings = Settings.from_env()
    assert settings.service_corpus_root == tmp_path
    assert settings.sandbox_fs_guard is False
    monkeypatch.setenv("RNSR_SANDBOX_FS_GUARD", "typo")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_secret_settings_are_not_in_repr():
    settings = Settings(service_token="do-not-log", trajectory_key="secret-key")
    assert "do-not-log" not in repr(settings)
    assert "secret-key" not in repr(settings)
