"""FMPClient contract tests — mocked HTTP, no API key required.

Pins the same guarantees FDClient's contract tests pin, plus the ones that
are specific to deriving fundamentals from statements:

1. Fail-loud: 401/402/5xx/network RAISE FMPClientError; 404 = no data.
2. Point-in-time: a ttm row is visible on end_date only once every
   statement behind it was FILED (filingDate), never on report_period.
3. ttm arithmetic: four contiguous quarters sum; margins/ROE/growth derive
   from those sums; a gap in the quarter chain yields no row.
4. Earnings: FMP's announcement calendar maps to 8-K records with BEAT/MISS
   from actual vs estimate; future rows without actuals are skipped.
5. ETFs: no statements → empty metrics (the snapshot then raises
   InsufficientData and LLM personas abstain, by design).
"""

from __future__ import annotations

import pytest
import requests

from hedge_fund.data import FMPClient, FMPClientError
from hedge_fund.data.provider import selected_provider


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else []
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("hedge_fund.data.fmp.time.sleep", lambda s: None)
    c = FMPClient(api_key="test-key")
    c._MIN_INTERVAL = 0.0
    yield c
    c.close()


def _route(client, table: dict[str, list | _FakeResponse | Exception]):
    """Serve responses by endpoint path; records every call's params."""
    calls = []

    def fake_request(method, url, **kwargs):
        path = url.replace(FMPClient.BASE_URL, "")
        calls.append({"path": path, **kwargs})
        r = table[path]
        if isinstance(r, list):
            r = r.pop(0) if r and isinstance(r[0], (_FakeResponse, Exception)) else _FakeResponse(200, r)
        if isinstance(r, Exception):
            raise r
        return r

    client._session.request = fake_request
    return calls


# ---------------------------------------------------------------------------
# Fixture statements: eight contiguous fiscal quarters, newest first.
# Revenue grows 100 per quarter; net income is 10% of revenue; 1,000 shares.
# ---------------------------------------------------------------------------

_QUARTER_ENDS = ["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30",
                 "2025-06-30", "2025-03-31", "2024-12-31", "2024-09-30"]
_FILED = ["2026-08-01", "2026-05-01", "2026-02-01", "2025-11-01",
          "2025-08-01", "2025-05-01", "2025-02-01", "2024-11-01"]


def _income():
    rows = []
    for i, (end, filed) in enumerate(zip(_QUARTER_ENDS, _FILED)):
        revenue = 1800.0 - 100 * i
        rows.append({
            "date": end, "filingDate": filed, "acceptedDate": f"{filed} 06:00:00",
            "period": f"Q{4 - (i % 4)}", "fiscalYear": end[:4], "reportedCurrency": "USD",
            "revenue": revenue, "grossProfit": revenue * 0.5, "operatingIncome": revenue * 0.2,
            "netIncome": revenue * 0.1, "ebitda": revenue * 0.25,
            "eps": revenue * 0.1 / 1000, "epsDiluted": revenue * 0.1 / 1000,
            "weightedAverageShsOut": 1000, "weightedAverageShsOutDil": 1000,
        })
    return rows


def _balance():
    return [{
        "date": end, "filingDate": filed, "acceptedDate": f"{filed} 06:00:00",
        "totalStockholdersEquity": 5000.0, "totalAssets": 10000.0, "totalDebt": 2000.0,
        "cashAndCashEquivalents": 500.0, "totalCurrentAssets": 3000.0, "totalCurrentLiabilities": 1500.0,
    } for end, filed in zip(_QUARTER_ENDS, _FILED)]


def _cashflow():
    return [{
        "date": end, "filingDate": filed, "freeCashFlow": 100.0, "operatingCashFlow": 150.0,
        "commonDividendsPaid": -20.0,
    } for end, filed in zip(_QUARTER_ENDS, _FILED)]


def _prices(close=50.0):
    """A close on the first weekday after each filing date (and the day itself)."""
    return [{"date": d, "open": close, "high": close, "low": close, "close": close, "volume": 1}
            for d in _FILED]


def _statement_routes(client, income=None, prices=None):
    return _route(client, {
        "/income-statement": income if income is not None else _income(),
        "/balance-sheet-statement": _balance(),
        "/cash-flow-statement": _cashflow(),
        "/historical-price-eod/full": prices if prices is not None else _prices(),
    })


# ---------------------------------------------------------------------------
# Fail-loud contract
# ---------------------------------------------------------------------------

def test_http_500_raises(client):
    _route(client, {"/historical-price-eod/full": _FakeResponse(500, text="boom")})
    with pytest.raises(FMPClientError) as exc_info:
        client.get_prices("AAPL", "2024-01-01", "2024-12-31")
    assert exc_info.value.status_code == 500


def test_http_401_raises(client):
    _route(client, {"/income-statement": _FakeResponse(401, text="bad key")})
    with pytest.raises(FMPClientError) as exc_info:
        client.get_financial_metrics("AAPL", "2026-09-30")
    assert exc_info.value.status_code == 401


def test_http_402_premium_endpoint_raises_not_empty(client):
    """A premium-locked endpoint must not read as 'no data'."""
    _route(client, {"/income-statement": _FakeResponse(402, text="Premium Query Parameter")})
    with pytest.raises(FMPClientError) as exc_info:
        client.get_financial_metrics("AAPL", "2026-09-30")
    assert exc_info.value.status_code == 402
    assert "premium" in str(exc_info.value).lower()


def test_network_error_raises(client):
    _route(client, {"/historical-price-eod/full": requests.ConnectionError("boom")})
    with pytest.raises(FMPClientError):
        client.get_prices("AAPL", "2024-01-01", "2024-12-31")


def test_404_means_no_data_not_failure(client):
    _route(client, {"/income-statement": _FakeResponse(404)})
    assert client.get_financial_metrics("ZZZZ", "2026-09-30") == []


def test_429_retries_then_raises_when_exhausted(client):
    _route(client, {"/historical-price-eod/full": [_FakeResponse(429)] * (len(FMPClient._RETRY_DELAYS) + 1)})
    with pytest.raises(FMPClientError) as exc_info:
        client.get_prices("AAPL", "2024-01-01", "2024-12-31")
    assert exc_info.value.status_code == 429


def test_error_body_with_200_raises(client):
    """FMP sometimes answers 200 with {"Error Message": ...}; that is a failure."""
    _route(client, {"/profile": _FakeResponse(200, {"Error Message": "Invalid API KEY"})})
    with pytest.raises(FMPClientError):
        client.get_company_facts("AAPL")


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def test_prices_map_fields_and_sort_ascending(client):
    _route(client, {"/historical-price-eod/full": [
        {"date": "2024-01-03", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 100},
        {"date": "2024-01-02", "open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 100},
    ]})
    prices = client.get_prices("AAPL", "2024-01-01", "2024-01-31")
    assert [p.time for p in prices] == ["2024-01-02", "2024-01-03"]
    assert prices[1].close == 2.0


def test_prices_walk_back_when_the_span_is_capped(client):
    """One response may not cover the window; the client keeps asking for
    the stretch before the earliest bar until the window is covered."""
    calls = _route(client, {"/historical-price-eod/full": [
        _FakeResponse(200, [{"date": "2024-06-03", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}]),
        _FakeResponse(200, [{"date": "2024-01-02", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}]),
    ]})
    prices = client.get_prices("AAPL", "2024-01-01", "2024-12-31")
    assert [p.time for p in prices] == ["2024-01-02", "2024-06-03"]
    assert calls[1]["params"]["to"] == "2024-06-02"


def test_prices_clip_to_window(client):
    _route(client, {"/historical-price-eod/full": [
        {"date": "2024-02-01", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"date": "2024-01-15", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
    ]})
    prices = client.get_prices("AAPL", "2024-01-10", "2024-01-31")
    assert [p.time for p in prices] == ["2024-01-15"]


# ---------------------------------------------------------------------------
# Point-in-time + ttm derivation
# ---------------------------------------------------------------------------

def test_metrics_are_visible_only_after_filing(client):
    """Q2-2026 ended 06-30 but filed 08-01: invisible on 07-15, visible on 08-01."""
    _statement_routes(client)
    before = client.get_financial_metrics("TEST", "2026-07-15", limit=10)
    assert before[0].report_period == "2026-03-31"
    assert all(m.filing_date <= "2026-07-15" for m in before)

    after = client.get_financial_metrics("TEST", "2026-08-01", limit=10)
    assert after[0].report_period == "2026-06-30"
    assert after[0].filing_date == "2026-08-01"


def test_ttm_sums_four_quarters(client):
    _statement_routes(client)
    rows = client.get_financial_metrics("TEST", "2026-12-31", limit=10)
    latest = rows[0]
    # revenues 1800+1700+1600+1500; net income 10% of that
    assert latest.earnings_per_share == pytest.approx(660.0 / 1000)
    assert latest.net_margin == pytest.approx(0.10)
    assert latest.gross_margin == pytest.approx(0.50)
    assert latest.return_on_equity == pytest.approx(660.0 / 5000)
    assert latest.debt_to_equity == pytest.approx(2000 / 5000)
    assert latest.current_ratio == pytest.approx(2.0)
    assert latest.book_value_per_share == pytest.approx(5.0)
    assert latest.free_cash_flow_per_share == pytest.approx(400.0 / 1000)
    assert latest.payout_ratio == pytest.approx(80.0 / 660.0)
    # market cap = 1000 shares × 50 close at the filing date; P/E = mc / ttm NI
    assert latest.market_cap == pytest.approx(50_000.0)
    assert latest.price_to_earnings_ratio == pytest.approx(50_000.0 / 660.0)
    # growth vs the ttm ending four quarters earlier: 6600 vs 1400+1300+1200+1100
    assert latest.revenue_growth == pytest.approx(6600 / 5000 - 1)
    # eight quarters → five ttm rows (indices 0..4); rows 5-7 lack a full window
    assert len(rows) == 5
    assert rows[-1].revenue_growth is None


def test_gap_in_quarter_chain_yields_no_row(client):
    income = _income()
    del income[1]  # drop 2026-03-31: the latest window now spans a gap
    _statement_routes(client, income=income)
    rows = client.get_financial_metrics("TEST", "2026-12-31", limit=10)
    assert rows[0].report_period == "2025-12-31"


def test_metrics_only_ttm(client):
    with pytest.raises(ValueError):
        client.get_financial_metrics("TEST", "2026-12-31", period="annual")


def test_statements_fetched_once_per_ticker(client):
    calls = _statement_routes(client)
    client.get_financial_metrics("TEST", "2026-01-01")
    client.get_financial_metrics("TEST", "2026-12-31")
    statement_calls = [c for c in calls if c["path"].endswith("-statement")]
    assert len(statement_calls) == 3


def test_etf_without_statements_is_empty_not_error(client):
    _route(client, {"/income-statement": [], "/balance-sheet-statement": [], "/cash-flow-statement": []})
    assert client.get_financial_metrics("VOO", "2026-09-30") == []


# ---------------------------------------------------------------------------
# Earnings
# ---------------------------------------------------------------------------

def test_earnings_history_maps_announcements_to_8k_records(client):
    _route(client, {
        "/earnings": [
            {"date": "2026-11-02", "epsActual": None, "epsEstimated": 1.99},          # future: skipped
            {"date": "2026-07-30", "epsActual": 2.02, "epsEstimated": 1.89,
             "revenueActual": 1800.0, "revenueEstimated": 1700.0},
            {"date": "2026-04-30", "epsActual": 1.80, "epsEstimated": 1.95},
            {"date": "2026-01-29", "epsActual": 2.00, "epsEstimated": 2.00},
        ],
        "/income-statement": _income(), "/balance-sheet-statement": _balance(), "/cash-flow-statement": _cashflow(),
    })
    history = client.get_earnings_history("TEST", limit=12)
    assert [r.filing_date for r in history] == ["2026-07-30", "2026-04-30", "2026-01-29"]
    assert all(r.source_type == "8-K" for r in history)
    assert [r.quarterly.eps_surprise for r in history] == ["BEAT", "MISS", "MEET"]
    assert history[0].quarterly.revenue_surprise == "BEAT"
    # the 07-30 announcement belongs to the quarter filed next (08-01 → 2026-06-30)
    assert history[0].report_period == "2026-06-30"
    assert (history[0].filing_date and history[0].report_period) and \
        (0 <= (__import__("datetime").date.fromisoformat(history[0].filing_date)
               - __import__("datetime").date.fromisoformat(history[0].report_period)).days < 45)


def test_get_earnings_is_latest_announced(client):
    _route(client, {
        "/earnings": [{"date": "2026-07-30", "epsActual": 2.02, "epsEstimated": 1.89}],
        "/income-statement": _income(), "/balance-sheet-statement": _balance(), "/cash-flow-statement": _cashflow(),
    })
    e = client.get_earnings("TEST")
    assert e is not None and e.quarterly.eps_surprise == "BEAT"


# ---------------------------------------------------------------------------
# Company facts, market cap
# ---------------------------------------------------------------------------

def test_company_facts_flags_etfs(client):
    _route(client, {"/profile": [{"symbol": "VOO", "companyName": "Vanguard S&P 500 ETF",
                                  "isEtf": True, "isActivelyTrading": True, "sector": "Financial Services"}]})
    facts = client.get_company_facts("VOO")
    assert facts is not None and facts.category == "ETF" and facts.name.startswith("Vanguard")


def test_market_cap_never_looks_ahead(client):
    calls = _route(client, {"/historical-market-capitalization": [
        {"date": "2026-10-09", "marketCap": 3.0}, {"date": "2026-10-07", "marketCap": 2.0},
    ]})
    assert client.get_market_cap("AAPL", "2026-10-08") == 2.0
    assert calls[0]["params"]["to"] == "2026-10-08"


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

def test_provider_selection(monkeypatch):
    monkeypatch.delenv("HEDGE_FUND_DATA_PROVIDER", raising=False)
    monkeypatch.delenv("FINANCIAL_DATASETS_API_KEY", raising=False)
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(ValueError):
        selected_provider()
    monkeypatch.setenv("FMP_API_KEY", "x")
    assert selected_provider() == "fmp"
    monkeypatch.setenv("FINANCIAL_DATASETS_API_KEY", "y")
    assert selected_provider() == "fd"            # FD wins when both are set
    monkeypatch.setenv("HEDGE_FUND_DATA_PROVIDER", "fmp")
    assert selected_provider() == "fmp"           # explicit override wins
    monkeypatch.setenv("HEDGE_FUND_DATA_PROVIDER", "bogus")
    with pytest.raises(ValueError):
        selected_provider()
