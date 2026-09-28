"""Optional query caps never disable accounting or erase attempted work."""

import asyncio
import json
import math

import pytest

from rnsr.config import Settings
from rnsr.errors import BudgetExhausted
from rnsr.harness.budget import BudgetLedger
from rnsr.llm.base import Usage


def test_default_query_crosses_all_former_caps_and_keeps_finite_accounting(monkeypatch):
    ledger = BudgetLedger.from_settings(Settings())
    started = ledger._t0
    monkeypatch.setattr("rnsr.harness.budget.time.monotonic", lambda: started + 7200)
    ledger.root_iters = 200
    for _ in range(1000):
        ledger.reserve_sub_call()
        ledger.add_usage(Usage(input_tokens=100, output_tokens=10, cost_usd=0.1))

    assert ledger.uncapped
    assert ledger.breached() is None
    assert ledger.remaining_root_iters() is None
    assert ledger.remaining_sub_calls() is None
    assert ledger.deadline() is None
    assert ledger.remaining_wall_s() == math.inf
    snapshot = ledger.snapshot()
    assert snapshot == {
        "root_iters": 200, "sub_calls": 1000, "wall_s": 7200.0,
        "spend_usd": 100.0, "input_tokens": 100000, "output_tokens": 10000,
    }
    # Unlimited allowance must never appear as non-standard Infinity in records.
    assert json.loads(json.dumps(snapshot, allow_nan=False)) == snapshot


@pytest.mark.parametrize("cap,usage", [
    ("max_root_iters", "root_iters"),
    ("max_sub_calls", "sub_calls"),
    ("max_spend_usd", "spend_usd"),
])
def test_positive_caps_are_enforced_at_boundary_and_zero_disables_them(cap, usage):
    ledger = BudgetLedger(**{cap: 3})
    assert not ledger.uncapped
    setattr(ledger, usage, 2)
    assert not ledger.limit_reached(cap)
    assert ledger.breached() is None
    setattr(ledger, usage, 3)
    assert ledger.limit_reached(cap)
    assert ledger.breached() == cap
    setattr(ledger, cap, 0)
    assert ledger.uncapped
    assert not ledger.limit_reached(cap)
    assert ledger.breached() is None


def test_optional_wall_deadline_and_remaining_time(monkeypatch):
    ledger = BudgetLedger(max_wall_s=10, _t0=100)
    assert not ledger.uncapped
    monkeypatch.setattr("rnsr.harness.budget.time.monotonic", lambda: 107)
    assert ledger.deadline() == 110
    assert ledger.remaining_wall_s() == 3
    assert ledger.breached() is None
    monkeypatch.setattr("rnsr.harness.budget.time.monotonic", lambda: 110)
    assert ledger.remaining_wall_s() == 0
    assert ledger.breached() == "max_wall_s"
    monkeypatch.setattr("rnsr.harness.budget.time.monotonic", lambda: 120)
    assert ledger.remaining_wall_s() == 0
    ledger.max_wall_s = 0
    assert ledger.uncapped
    assert ledger.deadline() is None
    assert ledger.remaining_wall_s() == math.inf
    assert ledger.breached() is None


def test_remaining_call_and_iteration_counts_are_nonnegative():
    ledger = BudgetLedger(max_root_iters=3, max_sub_calls=4)
    ledger.root_iters = ledger.sub_calls = 2
    assert ledger.remaining_root_iters() == 1
    assert ledger.remaining_sub_calls() == 2
    ledger.root_iters = ledger.sub_calls = 10
    assert ledger.remaining_root_iters() == ledger.remaining_sub_calls() == 0


def test_attempt_reservation_does_not_double_count_success_usage():
    ledger = BudgetLedger(max_sub_calls=2)
    ledger.reserve_sub_call()
    ledger.add_usage(Usage(input_tokens=7, output_tokens=3, cost_usd=0.05))
    assert ledger.sub_calls == 1
    assert ledger.usage == Usage(input_tokens=7, output_tokens=3, cost_usd=0.05)
    ledger.reserve_sub_call()
    with pytest.raises(BudgetExhausted) as exc:
        ledger.reserve_sub_call()
    assert (exc.value.cap, exc.value.limit, exc.value.spent) == ("max_sub_calls", 2, 2)
    assert ledger.sub_calls == 2


@pytest.mark.parametrize("cap", ["max_wall_s", "max_spend_usd"])
def test_other_explicit_caps_reject_attempt_without_charging_it(monkeypatch, cap):
    ledger = BudgetLedger(**{cap: 2}, _t0=100)
    monkeypatch.setattr("rnsr.harness.budget.time.monotonic", lambda: 102)
    ledger.spend_usd = 2
    with pytest.raises(BudgetExhausted) as exc:
        ledger.reserve_sub_call()
    assert exc.value.cap == cap
    assert ledger.sub_calls == 0


async def test_cancelled_in_flight_call_keeps_attempt_accounting():
    ledger = BudgetLedger()
    dispatched = asyncio.Event()

    async def request():
        ledger.reserve_sub_call()
        dispatched.set()
        await asyncio.Event().wait()

    pending = asyncio.create_task(request())
    await dispatched.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert ledger.sub_calls == 1
    assert ledger.spend_usd == 0
    assert ledger.breached() is None


def test_unknown_cap_is_rejected():
    with pytest.raises(ValueError, match="unknown query budget cap"):
        BudgetLedger().limit_reached("max_spned_usd")
