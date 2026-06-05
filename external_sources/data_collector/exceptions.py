class DataCollectorError(Exception):
    """Base exception"""
    pass

class CollectorNotFoundError(DataCollectorError):
    """collector پیدا نشد"""
    pass

class RateLimitError(DataCollectorError):
    """Rate limit خورد"""
    def __init__(self, source: str, retry_after: int = 60):
        self.source = source
        self.retry_after = retry_after
        super().__init__(f"Rate limit on {source}. Retry after {retry_after}s")

class AuthenticationError(DataCollectorError):
    """مشکل احراز هویت"""
    pass

class CollectionError(DataCollectorError):
    """خطای عمومی جمع‌آوری"""
    def __init__(self, source: str, reason: str):
        self.source = source
        super().__init__(f"[{source}] {reason}")
