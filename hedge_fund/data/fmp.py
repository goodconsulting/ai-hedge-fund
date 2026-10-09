"""Financial Modeling Prep (FMP) data client — a second DataClient provider.

Speaks FMP's ``/stable/`` API (the legacy ``/api/v3`` paths are retired for
new accounts) and returns exactly the models the Financial Datasets client
returns, so every analyst, the snapshot builder and the pipeline run
unchanged. Select it with ``HEDGE_FUND_DATA_PROVIDER=fmp`` or by having
``FMP_API_KEY`` set and no Financial Datasets key (see provider.py).

Why fundamentals are COMPUTED here rather than fetched: FMP's ready-made
``/ratios`` and ``/key-metrics`` endpoints carry a fiscal ``date`` but no
filing date, and their quarterly series is a premium tier. The quarterly
statements (income, balance sheet, cash flow) are on the base plan and carry
``filingDate``/``acceptedDate``, so this client derives trailing-twelve-month
metrics from them and stamps each row with the date it became public. That is
what makes ``get_financial_metrics`` honestly point-in-time: a row exists on
*end_date* only if all three statements behind it were filed by then.

Contract (identical to FDClient):
- empty list / None = the data genuinely does not exist (ETFs, unknown tickers)
- infrastructure failures RAISE FMPClientError (auth 401, premium 402,
  rate-limit 429 after retries, 5xx, network) — never a silent empty.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date as _date
from datetime import timedelta
from math import isfinite

import requests

from hedge_fund.data.models import (
    CompanyFacts,
    CompanyNews,
    Earnings,
    EarningsData,
    EarningsRecord,
    FinancialMetrics,
    InsiderTrade,
    Price,
)

logger = logging.getLogger(__name__)

# Quarters of statements fetched per ticker. 40 quarters = 10 years, enough
# for a 20-period ttm snapshot plus the four-quarter growth lookback.
_STATEMENT_QUARTERS = 40
# A filing is "priced" at the first close on or after the filing date,
# searched this many calendar days forward (holiday clusters).
_MARK_SEARCH_DAYS = 7
# An earnings announcement (8-K) precedes the matching 10-Q/10-K by at most
# this many days; used to pin a report_period onto an announcement date.
_ANNOUNCEMENT_TO_FILING_DAYS = 60
_FISCAL_QUARTER_DAYS = 100


class FMPClientError(Exception):
    """An API request failed for infrastructure reasons. Distinct from
    "no data exists" — that returns empty. A backtest must crash on this."""

    def __init__(self, message: str, *, status_code: int | None = None, path: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.path = path


class FMPClient:
    """DataClient over FMP ``/stable``.

    Usage::

        with FMPClient() as fmp:
            prices = fmp.get_prices("AAPL", "2024-01-01", "2024-12-31")

    Statements and the price series are fetched once per ticker per process
    and filtered locally for every as-of date, so a backtest costs one set
    of calls per name rather than one per session.
    """

    BASE_URL = "https://financialmodelingprep.com/stable"
    _RETRY_DELAYS = (5, 15, 30)
    _MIN_INTERVAL = 0.2  # 300 req/min on the standard plan

    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        self._api_key = api_key or os.environ.get("FMP_API_KEY", "")
        self._timeout = timeout
        self._session = requests.Session()
        self._last_request = 0.0
        self._statements: dict[str, tuple[list[dict], list[dict], list[dict]]] = {}
        self._price_series: dict[str, dict[str, Price]] = {}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> FMPClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        self._session.close()

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def get_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "day",
        interval_multiplier: int = 1,
    ) -> list[Price]:
        """Daily OHLCV bars in [start_date, end_date], oldest first."""
        if interval != "day" or interval_multiplier != 1:
            raise ValueError("FMPClient supports daily bars only")
        rows = self._get("/historical-price-eod/full", {
            "symbol": ticker, "from": start_date, "to": end_date,
        })
        bars = [_price(row) for row in rows or []]
        # FMP caps the span of one response; walk backwards until the window
        # is covered or the feed runs dry.
        while bars and min(b.time for b in bars) > start_date:
            earliest = min(b.time for b in bars)
            before = (_date.fromisoformat(earliest) - timedelta(days=1)).isoformat()
            more = self._get("/historical-price-eod/full", {
                "symbol": ticker, "from": start_date, "to": before,
            })
            older = [_price(row) for row in more or [] if str(row["date"])[:10] < earliest]
            if not older:
                break  # the feed has nothing before `earliest`: stop, never spin
            bars.extend(older)
        seen: dict[str, Price] = {}
        for bar in bars:
            seen.setdefault(bar.time, bar)
        return [seen[day] for day in sorted(seen) if start_date <= day <= end_date]

    # ------------------------------------------------------------------
    # Financial metrics (computed ttm, point-in-time)
    # ------------------------------------------------------------------

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[FinancialMetrics]:
        """Trailing-twelve-month metrics PUBLIC as of *end_date*, newest first.

        Each row is built from four consecutive quarterly statements and
        carries the filing date of the latest of them; rows filed after
        *end_date* are excluded. Only ``period="ttm"`` is supported — the
        pipeline uses nothing else, and FMP's own annual/quarterly ratios
        carry no filing date.
        """
        if period != "ttm":
            raise ValueError("FMPClient computes ttm metrics only; annual/quarterly are not point-in-time on FMP")
        income, balance, cashflow = self._quarterly_statements(ticker)
        if not income:
            return []
        rows = self._ttm_rows(ticker, income, balance, cashflow)
        public = [r for r in rows if r.filing_date is not None and r.filing_date <= end_date]
        return public[:limit]

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def get_news(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[CompanyNews]:
        params: dict = {"symbols": ticker, "to": end_date, "limit": min(limit, 250)}
        if start_date is not None:
            params["from"] = start_date
        rows = self._get("/news/stock", params) or []
        out: list[CompanyNews] = []
        for row in rows:
            published = str(row.get("publishedDate") or "")[:10]
            if published and published > end_date:
                continue
            if start_date is not None and published and published < start_date:
                continue
            out.append(CompanyNews(
                ticker=ticker, title=row.get("title") or "", source=row.get("publisher") or row.get("site") or "",
                date=row.get("publishedDate"), url=row.get("url"),
            ))
        return out[:limit]

    # ------------------------------------------------------------------
    # Insider trades
    # ------------------------------------------------------------------

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        out: list[InsiderTrade] = []
        page = 0
        page_size = 100
        while len(out) < limit:
            rows = self._get("/insider-trading/search", {
                "symbol": ticker, "page": page, "limit": page_size,
            }) or []
            for row in rows:
                filed = str(row.get("filingDate") or "")[:10]
                if not filed or filed > end_date:
                    continue
                if start_date is not None and filed < start_date:
                    continue
                shares = _float(row.get("securitiesTransacted"))
                price = _float(row.get("price"))
                out.append(InsiderTrade(
                    ticker=ticker,
                    name=row.get("reportingName") or "",
                    filing_date=filed,
                    issuer=None,
                    title=row.get("typeOfOwner"),
                    transaction_date=row.get("transactionDate"),
                    transaction_type=row.get("transactionType"),
                    transaction_shares=shares,
                    transaction_price_per_share=price,
                    transaction_value=shares * price if shares is not None and price is not None else None,
                    shares_owned_after_transaction=_float(row.get("securitiesOwned")),
                    security_title=row.get("securityName"),
                ))
            if len(rows) < page_size:
                break
            page += 1
        return out[:limit]

    # ------------------------------------------------------------------
    # Company facts
    # ------------------------------------------------------------------

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        rows = self._get("/profile", {"symbol": ticker})
        if not rows:
            return None
        p = rows[0]
        category = "ETF" if p.get("isEtf") else ("Fund" if p.get("isFund") else None)
        return CompanyFacts(
            ticker=ticker,
            is_active=bool(p.get("isActivelyTrading", True)),
            name=p.get("companyName"),
            cik=p.get("cik"),
            sector=p.get("sector"),
            industry=p.get("industry"),
            category=category,
            exchange=p.get("exchange"),
            location=", ".join(x for x in (p.get("city"), p.get("state"), p.get("country")) if x) or None,
            market_cap=_float(p.get("marketCap")),
        )

    # ------------------------------------------------------------------
    # Earnings
    # ------------------------------------------------------------------

    def get_earnings(self, ticker: str) -> Earnings | None:
        history = self.get_earnings_history(ticker, limit=1)
        if not history:
            return None
        r = history[0]
        return Earnings(ticker=ticker, report_period=r.report_period, fiscal_period=r.fiscal_period,
                        currency=r.currency, quarterly=r.quarterly, annual=None)

    def get_earnings_history(self, ticker: str, limit: int = 12) -> list[EarningsRecord]:
        """Announced quarters as 8-K style records, newest first.

        FMP's ``/earnings`` is the announcement calendar: ``date`` is the day
        the numbers were released (the event PEAD keys on), and future rows
        have no actuals and are skipped. The fiscal ``report_period`` is
        recovered from the quarterly income statement filed next.
        """
        rows = self._get("/earnings", {"symbol": ticker, "limit": max(limit * 2, 8)}) or []
        income, _, _ = self._quarterly_statements(ticker)
        out: list[EarningsRecord] = []
        for row in rows:
            actual = _float(row.get("epsActual"))
            estimate = _float(row.get("epsEstimated"))
            announced = str(row.get("date") or "")[:10]
            if actual is None or not announced:
                continue
            statement = _statement_for_announcement(announced, income)
            rev_actual = _float(row.get("revenueActual"))
            rev_est = _float(row.get("revenueEstimated"))
            out.append(EarningsRecord(
                ticker=ticker,
                report_period=statement["date"] if statement else announced,
                source_type="8-K",
                filing_date=announced,
                fiscal_period=f"{statement['period']} {statement['fiscalYear']}" if statement else None,
                currency=statement.get("reportedCurrency") if statement else None,
                quarterly=EarningsData(
                    revenue=rev_actual, estimated_revenue=rev_est,
                    revenue_surprise=_surprise(rev_actual, rev_est),
                    earnings_per_share=actual, estimated_earnings_per_share=estimate,
                    eps_surprise=_surprise(actual, estimate),
                ),
            ))
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------
    # Market cap
    # ------------------------------------------------------------------

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        """Market cap at the last session on or before *end_date* (no lookahead)."""
        start = (_date.fromisoformat(end_date) - timedelta(days=_MARK_SEARCH_DAYS)).isoformat()
        rows = self._get("/historical-market-capitalization", {
            "symbol": ticker, "from": start, "to": end_date,
        }) or []
        dated = [(str(r.get("date"))[:10], _float(r.get("marketCap"))) for r in rows]
        dated = [(d, mc) for d, mc in dated if d <= end_date and mc is not None]
        return max(dated)[1] if dated else None

    # ------------------------------------------------------------------
    # Fundamentals derivation
    # ------------------------------------------------------------------

    def _quarterly_statements(self, ticker: str) -> tuple[list[dict], list[dict], list[dict]]:
        if ticker not in self._statements:
            params = {"symbol": ticker, "period": "quarter", "limit": _STATEMENT_QUARTERS}
            income = self._get("/income-statement", params) or []
            if not income:
                # No income statements (ETF, fund, unknown ticker): nothing to
                # derive from, so don't spend two more calls finding out.
                self._statements[ticker] = ([], [], [])
                return self._statements[ticker]
            fetched = [income]
            for path in ("/balance-sheet-statement", "/cash-flow-statement"):
                fetched.append(self._get(path, params) or [])
            self._statements[ticker] = tuple(  # type: ignore[assignment]
                sorted(rows, key=lambda r: str(r.get("date")), reverse=True) for rows in fetched
            )
        return self._statements[ticker]

    def _ttm_rows(self, ticker: str, income: list[dict], balance: list[dict], cashflow: list[dict]) -> list[FinancialMetrics]:
        by_date_bal = {str(r["date"]): r for r in balance}
        by_date_cf = {str(r["date"]): r for r in cashflow}
        rows: list[FinancialMetrics] = []
        ttm_cache: dict[int, dict] = {}

        def ttm(i: int) -> dict | None:
            if i in ttm_cache:
                return ttm_cache[i]
            window = income[i:i + 4]
            if len(window) < 4 or not _contiguous(window):
                ttm_cache[i] = None
                return None
            cfs = [by_date_cf.get(str(q["date"])) for q in window]
            agg = {
                "revenue": _sum(q.get("revenue") for q in window),
                "gross_profit": _sum(q.get("grossProfit") for q in window),
                "operating_income": _sum(q.get("operatingIncome") for q in window),
                "net_income": _sum(q.get("netIncome") for q in window),
                "ebitda": _sum(q.get("ebitda") for q in window),
                "eps": _sum(q.get("epsDiluted", q.get("eps")) for q in window),
                "fcf": _sum(c.get("freeCashFlow") for c in cfs) if all(cfs) else None,
                "ocf": _sum(c.get("operatingCashFlow") for c in cfs) if all(cfs) else None,
                "dividends": _sum(c.get("commonDividendsPaid") for c in cfs) if all(cfs) else None,
                "filed": max(str(q.get("filingDate") or "") for q in window) or None,
            }
            ttm_cache[i] = agg
            return agg

        for i, q in enumerate(income):
            agg = ttm(i)
            if agg is None:
                continue
            period_end = str(q["date"])
            bal = by_date_bal.get(period_end)
            cf = by_date_cf.get(period_end)
            filed_dates = [agg["filed"]] + [str(s.get("filingDate") or "") for s in (bal, cf) if s]
            filing_date = max(d for d in filed_dates if d) if any(filed_dates) else None
            shares = _float(q.get("weightedAverageShsOutDil")) or _float(q.get("weightedAverageShsOut"))
            equity = _float(bal.get("totalStockholdersEquity")) if bal else None
            assets = _float(bal.get("totalAssets")) if bal else None
            debt = _float(bal.get("totalDebt")) if bal else None
            cash = _float(bal.get("cashAndCashEquivalents")) if bal else None
            market_cap = self._market_cap_at(ticker, filing_date, shares) if filing_date else None
            prior = ttm(i + 4)
            rows.append(FinancialMetrics(
                ticker=ticker,
                report_period=period_end,
                period="ttm",
                currency=q.get("reportedCurrency"),
                filing_date=filing_date,
                filing_datetime=(bal or q).get("acceptedDate") if filing_date else None,
                market_cap=market_cap,
                enterprise_value=_add(market_cap, _sub(debt, cash)),
                price_to_earnings_ratio=_div(market_cap, agg["net_income"], positive_denominator=True),
                price_to_book_ratio=_div(market_cap, equity, positive_denominator=True),
                price_to_sales_ratio=_div(market_cap, agg["revenue"], positive_denominator=True),
                enterprise_value_to_ebitda_ratio=_div(_add(market_cap, _sub(debt, cash)), agg["ebitda"], positive_denominator=True),
                enterprise_value_to_revenue_ratio=_div(_add(market_cap, _sub(debt, cash)), agg["revenue"], positive_denominator=True),
                free_cash_flow_yield=_div(agg["fcf"], market_cap, positive_denominator=True),
                gross_margin=_div(agg["gross_profit"], agg["revenue"], positive_denominator=True),
                operating_margin=_div(agg["operating_income"], agg["revenue"], positive_denominator=True),
                net_margin=_div(agg["net_income"], agg["revenue"], positive_denominator=True),
                return_on_equity=_div(agg["net_income"], equity, positive_denominator=True),
                return_on_assets=_div(agg["net_income"], assets, positive_denominator=True),
                current_ratio=_div(_float(bal.get("totalCurrentAssets")) if bal else None,
                                   _float(bal.get("totalCurrentLiabilities")) if bal else None,
                                   positive_denominator=True),
                debt_to_equity=_div(debt, equity, positive_denominator=True),
                debt_to_assets=_div(debt, assets, positive_denominator=True),
                revenue_growth=_growth(agg["revenue"], prior["revenue"] if prior else None),
                earnings_growth=_growth(agg["net_income"], prior["net_income"] if prior else None),
                earnings_per_share_growth=_growth(agg["eps"], prior["eps"] if prior else None),
                free_cash_flow_growth=_growth(agg["fcf"], prior["fcf"] if prior else None),
                operating_income_growth=_growth(agg["operating_income"], prior["operating_income"] if prior else None),
                ebitda_growth=_growth(agg["ebitda"], prior["ebitda"] if prior else None),
                payout_ratio=_div(-agg["dividends"] if agg["dividends"] is not None else None, agg["net_income"], positive_denominator=True),
                earnings_per_share=agg["eps"],
                book_value_per_share=_div(equity, shares, positive_denominator=True),
                free_cash_flow_per_share=_div(agg["fcf"], shares, positive_denominator=True),
            ))
        return rows

    def _market_cap_at(self, ticker: str, filing_date: str, shares: float | None) -> float | None:
        """Diluted shares × first close on/after the filing date."""
        if shares is None or shares <= 0:
            return None
        end = (_date.fromisoformat(filing_date) + timedelta(days=_MARK_SEARCH_DAYS)).isoformat()
        series = self._price_series.setdefault(ticker, {})
        if not any(filing_date <= day <= end for day in series):
            for bar in self.get_prices(ticker, filing_date, end):
                series[bar.time] = bar
        days = sorted(day for day in series if filing_date <= day <= end)
        if not days:
            return None
        return series[days[0]].close * shares

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict) -> list[dict] | None:
        """GET a list endpoint. None for 404; raises FMPClientError otherwise."""
        resp = self._request("GET", path, params={**params, "apikey": self._api_key})
        if resp is None:
            return None
        body = resp.json()
        if isinstance(body, dict):
            if "Error Message" in body or "message" in body:
                raise FMPClientError(f"GET {path} returned an error body: {str(body)[:200]}", path=path)
            return [body]
        return body

    def _request(self, method: str, path: str, **kwargs) -> requests.Response | None:
        """Fail-loud HTTP with retry on 429. Returns None only for 404."""
        url = self.BASE_URL + path
        for attempt, delay in enumerate((*self._RETRY_DELAYS, None)):
            self._throttle()
            try:
                resp = self._session.request(method, url, timeout=self._timeout, **kwargs)
            except requests.RequestException as exc:
                raise FMPClientError(f"{method} {path} failed: {exc}", path=path) from exc

            if resp.status_code == 429 and delay is not None:
                logger.info("FMP rate limited (429), retrying in %ds (attempt %d/%d)",
                            delay, attempt + 1, len(self._RETRY_DELAYS))
                time.sleep(delay)
                continue
            if resp.status_code == 404:
                return None
            if resp.status_code == 402:
                raise FMPClientError(
                    f"{method} {path} is a premium FMP endpoint for this plan: {resp.text[:200]}",
                    status_code=402, path=path,
                )
            if resp.status_code >= 400:
                raise FMPClientError(
                    f"{method} {path} returned {resp.status_code}: {resp.text[:200]}",
                    status_code=resp.status_code, path=path,
                )
            return resp

        raise FMPClientError(
            f"{method} {path} rate limited (429) after {len(self._RETRY_DELAYS)} retries",
            status_code=429, path=path,
        )

    def _throttle(self) -> None:
        wait = self._MIN_INTERVAL - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price(row: dict) -> Price:
    return Price(
        open=float(row["open"]), close=float(row["close"]), high=float(row["high"]),
        low=float(row["low"]), volume=int(row.get("volume") or 0), time=str(row["date"])[:10],
    )


def _float(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if isfinite(f) else None


def _sum(values) -> float | None:
    floats = [_float(v) for v in values]
    if any(f is None for f in floats):
        return None
    return float(sum(floats))


def _div(num: float | None, den: float | None, *, positive_denominator: bool = False) -> float | None:
    if num is None or den is None or den == 0 or (positive_denominator and den <= 0):
        return None
    out = num / den
    return out if isfinite(out) else None


def _add(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a + b


def _sub(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def _growth(now: float | None, before: float | None) -> float | None:
    if now is None or before is None or before <= 0:
        return None
    return now / before - 1


def _surprise(actual: float | None, estimate: float | None) -> str | None:
    if actual is None or estimate is None:
        return None
    if actual > estimate:
        return "BEAT"
    if actual < estimate:
        return "MISS"
    return "MEET"


def _contiguous(window: list[dict]) -> bool:
    """Four quarters are a ttm only if each is ~one quarter apart."""
    dates = [_date.fromisoformat(str(q["date"])[:10]) for q in window]
    return all(0 < (a - b).days <= _FISCAL_QUARTER_DAYS * 1.4 for a, b in zip(dates, dates[1:]))


def _statement_for_announcement(announced: str, income: list[dict]) -> dict | None:
    """The quarterly statement an earnings release belongs to: the first one
    FILED on or after the announcement (within the window), else the latest
    fiscal quarter that ENDED before it."""
    limit = (_date.fromisoformat(announced) + timedelta(days=_ANNOUNCEMENT_TO_FILING_DAYS)).isoformat()
    filed_after = [q for q in income if announced <= str(q.get("filingDate") or "") <= limit]
    if filed_after:
        return min(filed_after, key=lambda q: str(q.get("filingDate")))
    ended_before = [q for q in income if str(q.get("date"))[:10] < announced]
    return max(ended_before, key=lambda q: str(q.get("date"))) if ended_before else None
