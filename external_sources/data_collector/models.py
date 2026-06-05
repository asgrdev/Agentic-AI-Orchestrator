from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Any
from enum import Enum


class DataDomain(str, Enum):
    KNOWLEDGE  = "knowledge"
    SCIENTIFIC = "scientific"
    NEWS       = "news"
    SOCIAL     = "social"
    FINANCIAL  = "financial"
    POLITICAL  = "political"
    BOOKS      = "books"


class DataSource(str, Enum):
    WIKIPEDIA        = "wikipedia"
    HUGGINGFACE      = "huggingface"
    ARXIV            = "arxiv"
    OPENALEX         = "openalex"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    NEWSPAPER        = "newspaper"
    RSS_FEED         = "rss_feed"
    REDDIT           = "reddit"
    TELEGRAM         = "telegram"
    BLUESKY          = "bluesky"
    YFINANCE         = "yfinance"
    FRED             = "fred"
    WORLD_BANK       = "world_bank"
    DATA_GOV         = "data_gov"
    EUROSTAT         = "eurostat"
    GUTENBERG        = "gutenberg"
    WEB_SCRAPE       = "web_scrape"


class CollectedItem(BaseModel):
    """مدل یکپارچه برای همه داده‌های جمع‌آوری شده"""

    id:          str
    title:       Optional[str]       = None
    content:     str
    summary:     Optional[str]       = None
    source:      DataSource
    domain:      DataDomain
    url:         Optional[str]       = None
    author:      Optional[str]       = None
    published_at: Optional[datetime] = None
    collected_at: datetime           = Field(default_factory=datetime.utcnow)
    language:    str                 = "en"
    tags:        list[str]           = []
    metadata:    dict[str, Any]      = {}
    score:       Optional[float]     = None   # relevance score

    class Config:
        use_enum_values = True


class CollectionResult(BaseModel):
    """نتیجه یک عملیات جمع‌آوری"""

    success:      bool
    source:       DataSource
    total:        int
    items:        list[CollectedItem]
    errors:       list[str]          = []
    elapsed_sec:  float              = 0.0
    query:        Optional[str]      = None


class SearchQuery(BaseModel):
    """مدل کوئری جستجو"""

    query:       str
    domain:      Optional[DataDomain]   = None
    sources:     list[DataSource]       = []
    max_results: int                    = 10
    language:    str                    = "en"
    date_from:   Optional[datetime]     = None
    date_to:     Optional[datetime]     = None
    filters:     dict[str, Any]         = {}
