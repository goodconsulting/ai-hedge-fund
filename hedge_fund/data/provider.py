"""Which data provider runs, and how every entry point opens it.

    with open_data_client() as data:      # a CachedDataClient over the provider
        ...

Selection, in order:
1. ``HEDGE_FUND_DATA_PROVIDER`` = ``fd`` | ``fmp`` (explicit; wins).
2. ``FINANCIAL_DATASETS_API_KEY`` set → Financial Datasets.
3. ``FMP_API_KEY`` set → Financial Modeling Prep.
4. Neither → ValueError naming both variables, the only thing a user can act on.

Each provider caches under its own directory (cache/data/<provider>/): the
two vendors' numbers differ (computed vs served ttm), and a cache that mixed
them would hand one provider's history to the other.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Literal

from hedge_fund.data.cached import CachedDataClient, DEFAULT_CACHE_DIR
from hedge_fund.data.client import FDClient
from hedge_fund.data.fmp import FMPClient

Provider = Literal["fd", "fmp"]

PROVIDER_ENV_VARS: dict[Provider, str] = {
    "fd": "FINANCIAL_DATASETS_API_KEY",
    "fmp": "FMP_API_KEY",
}
PROVIDER_NAMES: dict[Provider, str] = {
    "fd": "Financial Datasets",
    "fmp": "Financial Modeling Prep",
}


def selected_provider() -> Provider:
    """The provider the environment selects. Raises ValueError when none is usable."""
    explicit = os.environ.get("HEDGE_FUND_DATA_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in PROVIDER_ENV_VARS:
            raise ValueError(
                f"HEDGE_FUND_DATA_PROVIDER={explicit!r} is not one of {sorted(PROVIDER_ENV_VARS)}"
            )
        return explicit  # type: ignore[return-value]
    for provider, var in PROVIDER_ENV_VARS.items():
        if os.environ.get(var):
            return provider
    raise ValueError(
        "no data provider configured: set FINANCIAL_DATASETS_API_KEY (Financial Datasets) "
        "or FMP_API_KEY (Financial Modeling Prep), optionally HEDGE_FUND_DATA_PROVIDER=fd|fmp"
    )


def missing_data_key() -> str | None:
    """The env var the selected provider still needs, or None when ready."""
    try:
        provider = selected_provider()
    except ValueError:
        explicit = os.environ.get("HEDGE_FUND_DATA_PROVIDER", "").strip().lower()
        return PROVIDER_ENV_VARS.get(explicit, PROVIDER_ENV_VARS["fd"])  # type: ignore[arg-type]
    var = PROVIDER_ENV_VARS[provider]
    return None if os.environ.get(var) else var


def data_provider_label(env_var: str) -> str:
    """Human name for the provider a key variable belongs to (for prompts)."""
    for provider, var in PROVIDER_ENV_VARS.items():
        if var == env_var:
            return PROVIDER_NAMES[provider]
    return "data provider"


@contextmanager
def open_data_client(*, refresh: bool = False) -> Iterator[CachedDataClient]:
    """A disk-cached DataClient over the selected provider, closed on exit."""
    provider = selected_provider()
    raw = FDClient() if provider == "fd" else FMPClient()
    with raw as client:  # both clients close their HTTP session on exit
        yield CachedDataClient(client, cache_dir=DEFAULT_CACHE_DIR / provider, refresh=refresh)
