import yfinance as yf
import pandas as pd
from fredapi import Fred
import wbdata
from loguru import logger
from external_sources.data_collector.models import CollectedItem, CollectionResult, DataSource, DataDomain
from external_sources.data_collector.helper import DataHelper
from external_sources.data_collector.exceptions import AuthenticationError
import time
from external_sources.data_collector.timeout import with_timeout


class FinancialCollector:
    """جمع‌آوری داده‌های مالی: سهام، اقتصاد کلان، World Bank"""

    def __init__(self, fred_api_key: str | None = None):
        self._fred = Fred(api_key=fred_api_key) if fred_api_key else None

    # ─── Yahoo Finance ───────────────────────────────────────────────────
    @with_timeout(5.0)#20.0
    def get_stock_info(self, ticker: str) -> CollectionResult:
        """اطلاعات سهام + توضیح شرکت"""
        start = time.time()
        items, errors = [], []

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            description = info.get("longBusinessSummary", "")
            if description:
                items.append(CollectedItem(
                    id=DataHelper.make_id("yfinance", ticker),
                    title=f"{info.get('longName', ticker)} ({ticker})",
                    content=DataHelper.clean_text(description),
                    summary=DataHelper.truncate(description, 300),
                    source=DataSource.YFINANCE,
                    domain=DataDomain.FINANCIAL,
                    url=info.get("website", ""),
                    tags=[
                        info.get("sector", ""),
                        info.get("industry", ""),
                        ticker
                    ],
                    metadata={
                        "symbol":        ticker,
                        "sector":        info.get("sector"),
                        "industry":      info.get("industry"),
                        "market_cap":    info.get("marketCap"),
                        "pe_ratio":      info.get("trailingPE"),
                        "52w_high":      info.get("fiftyTwoWeekHigh"),
                        "52w_low":       info.get("fiftyTwoWeekLow"),
                        "currency":      info.get("currency"),
                        "country":       info.get("country"),
                    }
                ))

            # اخبار مرتبط
            for news_item in (stock.news or [])[:5]:
                items.append(CollectedItem(
                    id=DataHelper.make_id("yfinance_news", news_item.get("uuid", "")),
                    title=news_item.get("title", ""),
                    content=news_item.get("title", ""),
                    source=DataSource.YFINANCE,
                    domain=DataDomain.FINANCIAL,
                    url=news_item.get("link", ""),
                    published_at=DataHelper.normalize_date(
                        news_item.get("providerPublishTime")
                    ),
                    tags=[ticker],
                    metadata={"type": "news", "ticker": ticker}
                ))

        except Exception as e:
            errors.append(f"yfinance [{ticker}]: {e}")

        return CollectionResult(
            success=len(items) > 0,
            source=DataSource.YFINANCE,
            total=len(items),
            items=items,
            errors=errors,
            elapsed_sec=time.time() - start
        )

    def get_multiple_stocks(
        self,
        tickers: list[str]
    ) -> CollectionResult:
        """چند سهام با هم"""
        all_items, all_errors = [], []
        for ticker in tickers:
            result = self.get_stock_info(ticker)
            all_items.extend(result.items)
            all_errors.extend(result.errors)

        return CollectionResult(
            success=len(all_items) > 0,
            source=DataSource.YFINANCE,
            total=len(all_items),
            items=all_items,
            errors=all_errors,
            elapsed_sec=0
        )

    # ─── FRED ────────────────────────────────────────────────────────────
    
    @with_timeout(5)#20.0)
    def get_fred_series(
        self,
        series_ids: list[str],
    ) -> CollectionResult:
        """سری‌های اقتصادی از Federal Reserve"""
        start = time.time()
        items, errors = [], []

        if self._fred is None:
            raise AuthenticationError("FRED API key not provided")

        SERIES_NAMES = {
            "GDP":    "Gross Domestic Product",
            "CPIAUCSL": "Consumer Price Index (Inflation)",
            "UNRATE": "Unemployment Rate",
            "FEDFUNDS": "Federal Funds Rate",
            "DGS10":  "10-Year Treasury Rate",
        }

        for sid in series_ids:
            try:
                info   = self._fred.get_series_info(sid)
                series = self._fred.get_series(sid)

                latest_value = series.iloc[-1] if not series.empty else None
                summary = (
                    f"Series: {info.get('title', sid)}\n"
                    f"Latest value: {latest_value:.2f}\n"
                    f"Units: {info.get('units', '')}\n"
                    f"Frequency: {info.get('frequency', '')}"
                )

                items.append(CollectedItem(
                    id=DataHelper.make_id("fred", sid),
                    title=info.get("title", SERIES_NAMES.get(sid, sid)),
                    content=summary,
                    source=DataSource.FRED,
                    domain=DataDomain.FINANCIAL,
                    url=f"https://fred.stlouisfed.org/series/{sid}",
                    metadata={
                        "series_id":    sid,
                        "units":        info.get("units"),
                        "frequency":    info.get("frequency"),
                        "latest_value": float(latest_value) if latest_value else None,
                        "latest_date":  str(series.index[-1]) if not series.empty else None,
                    }
                ))
            except Exception as e:
                errors.append(f"FRED [{sid}]: {e}")

        return CollectionResult(
            success=len(items) > 0,
            source=DataSource.FRED,
            total=len(items),
            items=items,
            errors=errors,
            elapsed_sec=time.time() - start
        )
