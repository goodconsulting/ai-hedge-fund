"""target_allocation — a quant model that expresses a fixed-weight ETF policy.

Every other registered analyst reasons over SEC-filed fundamentals or EPS
surprises. ETFs (VOO, AVUV, VEA, SPEM, IBIT, ...) file neither, so a
retirement-style book of index funds would be permanently flat. This model
turns a target-allocation policy with rebalance bands into the same Signal
every analyst emits, so the blend, master risk, the ledger and the
backtester apply unchanged.

How a conviction becomes a book (hedge_fund/portfolio/construction.py): in a
long_only strategy, weight = max(conviction, 0) / Σ|scores| × gross_target.
The RELATIVE size of the convictions across the universe therefore decides
the book. So this model does not emit "drift"; it emits the book it wants:

- outside the band (any sleeve's |target − current| > band): conviction =
  the sleeve's TARGET weight → the blend reproduces the target allocation
  exactly and the fund rebalances to it, selling the over-weight sleeves.
- inside the band for every sleeve: conviction = the sleeve's CURRENT weight
  → the blend reproduces the book as held and no orders result (sub-share
  dust aside). A band without this would be decorative: the engine
  rebalances every rebalance session, not only when drift is large.

The current book comes from the fund's own ledger (``fund_dir``): the latest
SessionRecord's positions × marks / NAV. Without a ledger (a backtest, or
the first session) there is no "current" and the model always targets.
"""

from __future__ import annotations

from pathlib import Path

from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal
from hedge_fund.signals.base import QuantModel

_SUM_TOLERANCE = 1e-6


class TargetAllocationModel(QuantModel):
    """Fixed-weight allocation with rebalance bands, as an alpha model."""

    investment_approach = "long_only"

    def __init__(
        self,
        *,
        targets: dict[str, float] | None = None,
        band: float = 0.05,
        fund_dir: str | None = None,
    ) -> None:
        # Every registry entry must construct bare (the TUI and the registry
        # test do `cls()`); without targets the model abstains on every name.
        cleaned = {str(t).strip().upper(): float(w) for t, w in (targets or {}).items()}
        if cleaned and any(w <= 0 for w in cleaned.values()):
            raise ValueError("every target weight must be positive")
        if cleaned and abs(sum(cleaned.values()) - 1.0) > _SUM_TOLERANCE:
            raise ValueError(f"targets must sum to 1.0, got {sum(cleaned.values()):.6f}")
        if not 0 <= band < 1:
            raise ValueError("band must be in [0, 1)")
        self._targets = cleaned
        self._band = band
        self._fund_dir = Path(fund_dir).expanduser() if fund_dir else None

    @property
    def name(self) -> str:
        return "target_allocation"

    @property
    def targets(self) -> dict[str, float]:
        return dict(self._targets)

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        symbol = ticker.upper()
        target = self._targets.get(symbol)
        if target is None:
            return Signal(
                model_name=self.name, ticker=ticker, date=date, value=0.0,
                reasoning="abstained: not in the allocation targets" if self._targets else "abstained: no targets configured",
                metadata={"abstained": True, "abstain_reason": "not in targets"},
            )
        current = self.current_weights()
        held = current.get(symbol, 0.0)
        drift = target - held
        breach = current and any(
            abs(self._targets[t] - current.get(t, 0.0)) > self._band for t in self._targets
        )
        if not current:
            value, action = target, "no book on record: targeting"
        elif breach:
            value, action = target, "band breached: rebalancing to target"
        else:
            value, action = held, "inside band: holding"
        return Signal(
            model_name=self.name, ticker=ticker, date=date, value=value,
            reasoning=(f"{action} — target {target:.1%}, held {held:.1%}, "
                       f"drift {drift:+.1%}, band ±{self._band:.1%}"),
            metadata={"target": target, "current": held, "drift": drift,
                      "band": self._band, "rebalance": bool(breach or not current),
                      "abstained": False},
        )

    # ------------------------------------------------------------------
    # The current book
    # ------------------------------------------------------------------

    def current_weights(self) -> dict[str, float]:
        """Weights of the latest recorded session, or {} when there is none."""
        if self._fund_dir is None:
            return {}
        # Lazy: signals must not import the paper package at module import
        # time (paper → pipeline → fund → signals would be a cycle).
        from hedge_fund.paper.ledger import Ledger

        latest = Ledger(self._fund_dir).latest()
        if latest is None or latest.nav <= 0:
            return {}
        return {
            t: shares * latest.marks[t] / latest.nav
            for t, shares in latest.positions.items() if shares and t in latest.marks
        }
