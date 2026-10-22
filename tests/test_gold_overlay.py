"""The overlay accepts a gold's own justification, not anything else."""
import json

import pytest

from rnsr.eval.gold_overlay import DEFAULT_OVERLAY, GoldOverlay


def test_bundled_overlay_accepts_the_documented_equivalents_only():
    ov = GoldOverlay.load(DEFAULT_OVERLAY)
    assert ov.version.startswith("financebench-overlay")
    # 8,738 million stated by the gold in billions
    assert ov.accepts("financebench_id_04672", "3M's net PP&E was $8,738 million at year-end FY2018.")
    assert ov.accepts("financebench_id_04672", "approximately $8.74 billion")
    assert not ov.accepts("financebench_id_04672", "$9,100 million")
    # 381.6 million rounded by the gold to 382
    assert ov.accepts("financebench_id_07661", "Block generated approximately $381.6 million in operating cash flow")
    assert not ov.accepts("financebench_id_07661", "$3,816 million")
    # sign convention: both magnitudes for both years must be present
    assert ov.accepts("financebench_id_00585", "The rate fell from 14.8% in FY2021 to (0.6)% in FY2022.")
    assert not ov.accepts("financebench_id_00585", "The rate was (0.6)% in FY2022.")
    # golds that are wrong by their own definition are not overlaid
    assert not ov.accepts("financebench_id_10420", "-1.42%")
    assert not ov.accepts("financebench_id_00283", "70")
    assert not ov.accepts("unknown_qid", "anything")


def test_overlay_entries_must_carry_a_reason_and_source(tmp_path):
    bad = tmp_path / "o.json"
    bad.write_text(json.dumps({"version": "x", "entries": {"q": {"accepted": ["1"]}}}))
    with pytest.raises(ValueError, match="reason"):
        GoldOverlay.load(bad)
    empty = tmp_path / "e.json"
    empty.write_text(json.dumps({"version": "x", "entries": {"q": {"reason": "r", "source": "s"}}}))
    with pytest.raises(ValueError, match="accepts nothing"):
        GoldOverlay.load(empty)
