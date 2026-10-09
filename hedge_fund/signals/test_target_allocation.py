"""target_allocation — the ETF policy model, end to end through the blend."""

from __future__ import annotations

import pytest

from hedge_fund.fund import BlendPolicy, ModelSpec, StrategySpec
from hedge_fund.portfolio.construction import blend_signals
from hedge_fund.signals import ALPHA_MODEL_REGISTRY, get_investment_approach
from hedge_fund.signals.target_allocation import TargetAllocationModel

TARGETS = {"VOO": 0.55, "AVUV": 0.15, "VEA": 0.15, "SPEM": 0.10, "IBIT": 0.05}


class _NoData:
    """The model never touches the data client; anything satisfies it."""


def _blend(model, universe, as_of="2026-01-05"):
    signals = [model.predict(t, as_of, _NoData()) for t in universe]
    return blend_signals(
        signals, {"target_allocation": 1.0}, 1.0, mode="long_only",
        investment_approaches={"target_allocation": "long_only"},
    )


def test_registered_long_only():
    assert ALPHA_MODEL_REGISTRY["target_allocation"] is TargetAllocationModel
    assert get_investment_approach("target_allocation") == "long_only"
    ModelSpec(name="target_allocation", params={"targets": TARGETS})
    StrategySpec(name="etf", models=[ModelSpec(name="target_allocation", params={"targets": TARGETS})],
                 blend=BlendPolicy(mode="long_only"))


def test_targets_must_be_a_full_positive_allocation():
    with pytest.raises(ValueError):
        TargetAllocationModel(targets={"VOO": 0.6, "VEA": 0.3})
    with pytest.raises(ValueError):
        TargetAllocationModel(targets={"VOO": 1.2, "VEA": -0.2})
    bare = TargetAllocationModel()  # registry contract: constructs with no args, abstains
    assert bare.predict("VOO", "2026-01-05", _NoData()).metadata["abstained"] is True
    with pytest.raises(ValueError):
        TargetAllocationModel(targets=TARGETS, band=1.0)


def test_without_a_book_the_blend_reproduces_the_targets():
    model = TargetAllocationModel(targets=TARGETS)
    result = _blend(model, list(TARGETS))
    for ticker, weight in TARGETS.items():
        assert result.weights[ticker] == pytest.approx(weight)


def test_ticker_outside_targets_abstains_and_gets_no_weight():
    model = TargetAllocationModel(targets=TARGETS)
    signal = model.predict("TSLA", "2026-01-05", _NoData())
    assert signal.metadata["abstained"] is True and signal.value == 0.0
    result = _blend(model, [*TARGETS, "TSLA"])
    assert result.weights["TSLA"] == 0.0
    assert result.weights["VOO"] == pytest.approx(0.55)


def test_inside_band_holds_the_book_as_held(monkeypatch, tmp_path):
    model = TargetAllocationModel(targets=TARGETS, band=0.05, fund_dir=tmp_path)
    held = {"VOO": 0.57, "AVUV": 0.14, "VEA": 0.15, "SPEM": 0.09, "IBIT": 0.05}
    monkeypatch.setattr(model, "current_weights", lambda: held)
    signals = {t: model.predict(t, "2026-01-05", _NoData()) for t in TARGETS}
    assert all(s.metadata["rebalance"] is False for s in signals.values())
    result = _blend(model, list(TARGETS))
    for ticker, weight in held.items():
        assert result.weights[ticker] == pytest.approx(weight)


def test_band_breach_on_one_sleeve_rebalances_every_sleeve(monkeypatch, tmp_path):
    model = TargetAllocationModel(targets=TARGETS, band=0.05, fund_dir=tmp_path)
    held = {"VOO": 0.62, "AVUV": 0.12, "VEA": 0.13, "SPEM": 0.08, "IBIT": 0.05}  # VOO +7pts
    monkeypatch.setattr(model, "current_weights", lambda: held)
    signals = {t: model.predict(t, "2026-01-05", _NoData()) for t in TARGETS}
    assert all(s.metadata["rebalance"] is True for s in signals.values())
    result = _blend(model, list(TARGETS))
    for ticker, weight in TARGETS.items():
        assert result.weights[ticker] == pytest.approx(weight)
    assert "band breached" in signals["AVUV"].reasoning


def test_missing_sleeve_counts_as_full_drift(monkeypatch, tmp_path):
    model = TargetAllocationModel(targets=TARGETS, band=0.04, fund_dir=tmp_path)
    # every held sleeve sits on target; only IBIT (5% target) is absent → 5pt drift > 4pt band
    monkeypatch.setattr(model, "current_weights", lambda: {"VOO": 0.55, "AVUV": 0.15, "VEA": 0.15, "SPEM": 0.10})
    signal = model.predict("IBIT", "2026-01-05", _NoData())
    assert signal.metadata["rebalance"] is True and signal.value == pytest.approx(0.05)


def test_current_weights_come_from_the_ledger(tmp_path):
    from hedge_fund.paper.ledger import Ledger
    from hedge_fund.pipeline.session import SessionRecord

    ledger = Ledger(tmp_path)
    ledger.init()
    record = SessionRecord(
        fund="etf", session="2026-01-05", universe=["VOO", "VEA"], nav=1000.0, cash=100.0,
        positions={"VOO": 6, "VEA": 3}, marks={"VOO": 100.0, "VEA": 100.0}, benchmark="VOO",
        benchmark_close=100.0, rebalance=True, code_version="test", mandate_hash="x",
    )
    record.hash = record.compute_hash()
    ledger.append(record)

    model = TargetAllocationModel(targets={"VOO": 0.7, "VEA": 0.3}, fund_dir=tmp_path)
    assert model.current_weights() == {"VOO": pytest.approx(0.6), "VEA": pytest.approx(0.3)}
    assert TargetAllocationModel(targets={"VOO": 1.0}).current_weights() == {}
