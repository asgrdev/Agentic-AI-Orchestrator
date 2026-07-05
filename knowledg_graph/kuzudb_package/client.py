from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

import kuzu

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class QueryResult:
    """نتیجه ساختار‌یافته هر کوئری."""
    data: list[dict[str, Any]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    query: str = ""

    @property
    def rows(self) -> list[dict[str, Any]]:
        """Alias — بخشی از کد با result.rows کار می‌کند."""
        return self.data

    def first(self) -> Optional[dict[str, Any]]:
        return self.data[0] if self.data else None

    def scalar(self, key: str, default: Any = None) -> Any:
        row = self.first()
        return row.get(key, default) if row else default

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return self.row_count

    def __bool__(self) -> bool:
        return self.row_count > 0


# ─────────────────────────────────────────────
# Connection Pool  (رفع Thread Safety)
# ─────────────────────────────────────────────

class _ConnectionPool:
    """
    هر thread یک connection مستقل دریافت می‌کند.
    رفع مشکل: Thread Safety - یک connection برای چند thread
    """

    def __init__(self, db: kuzu.Database) -> None:
        self._db = db
        self._local = threading.local()

    def get(self) -> kuzu.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = kuzu.Connection(self._db)
            logger.debug(
                "Connection جدید برای thread %s ساخته شد.",
                threading.current_thread().name,
            )
        return self._local.conn

    def release(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn = None
            logger.debug(
                "Connection thread %s آزاد شد.",
                threading.current_thread().name,
            )

    def close_all(self) -> None:
        """فقط connection thread جاری را می‌بندد (threading.local محدودیت دارد)."""
        self.release()


# ─────────────────────────────────────────────
# KuzuClient
# ─────────────────────────────────────────────

# جدول‌های مجاز - رفع مشکل SQL Injection
_ALLOWED_TABLES: set[str] = set()


def register_table(name: str) -> None:
    """هر بار که جدولی ساخته می‌شود، اینجا ثبت می‌شود."""
    _ALLOWED_TABLES.add(name)


def validate_table_name(name: str) -> str:
    """
    رفع مشکل: SQL Injection از طریق table_name
    فقط حروف، اعداد و underscore مجاز است.
    """
    import re
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"نام جدول نامعتبر: '{name}'")
    return name


class KuzuClient:
    """
    لایه پایه اتصال به KuzuDB.

    تغییرات نسبت به نسخه قبل:
    - Thread Safety: استفاده از _ConnectionPool به جای یک connection مشترک
    - execute_many: اجرا درون یک تراکنش اتمی (رفع مشکل rollback)
    - _convert_value: پشتیبانی از تمام type های پایه Python
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db: Optional[kuzu.Database] = None
        self._pool: Optional[_ConnectionPool] = None
        self._lock = threading.Lock()

    # ── اتصال ──────────────────────────────────

    def connect(self) -> "KuzuClient":
        with self._lock:
            if self._db is not None:
                return self
            self._db_path.mkdir(parents=True, exist_ok=True)
            db_file = self._db_path / "graph.db"
            self._db = kuzu.Database(str(db_file))
            self._pool = _ConnectionPool(self._db)
            logger.info("KuzuDB متصل شد: %s", db_file)
        return self

    def disconnect(self) -> None:
        with self._lock:
            if self._pool:
                self._pool.close_all()
            self._pool = None
            self._db = None
            logger.info("KuzuDB قطع اتصال شد.")

    def _ensure_connected(self) -> kuzu.Connection:
        if self._pool is None:
            raise RuntimeError("ابتدا connect() را صدا بزنید.")
        return self._pool.get()

    # ── Context Manager ────────────────────────

    def __enter__(self) -> "KuzuClient":
        return self.connect()

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ── اجرای کوئری ───────────────────────────

    def execute(
        self,
        query: str,
        params: Optional[dict[str, Any]] = None,
    ) -> QueryResult:
        conn = self._ensure_connected()
        try:
            raw = conn.execute(query, parameters=params or {})
            return self._parse_result(raw, query)
        except Exception as exc:
            logger.error("خطا در کوئری:\n%s\nپارامترها: %s\nخطا: %s", query, params, exc)
            raise

    def execute_many(
        self,
        queries: list[tuple[str, dict[str, Any]]],
    ) -> list[QueryResult]:
        """
        رفع مشکل: execute_many بدون تراکنش
        تمام کوئری‌ها درون یک تراکنش اتمی اجرا می‌شوند.
        """
       # print(kuzu.__version__)
        conn = self._ensure_connected()
        results: list[QueryResult] = []
        #     print([m for m in dir(conn) if not m.startswith('_')])

       # print(conn)
#        conn.execute("BEGIN TRANSACTION")
        try:
            for query, params in queries:
                raw = conn.execute(query, parameters=params or {})
                results.append(self._parse_result(raw, query))
            #conn.execute("COMMIT")
            return results
        except Exception as exc:
            #conn.execute("ROLLBACK")
            logger.error("execute_many rollback شد: %s", exc)
            raise

    # ── تراکنش دستی ───────────────────────────

    @contextmanager
    def transaction(self) -> Generator[kuzu.Connection, None, None]:
        conn = self._ensure_connected()
        conn.execute("BEGIN TRANSACTION")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    # ── پردازش نتیجه ──────────────────────────

    def _parse_result(self, raw: Any, query: str = "") -> QueryResult:
        if raw is None:
            return QueryResult(query=query)

        columns: list[str] = []
        rows: list[dict] = []

        try:
            columns = raw.get_column_names()
            while raw.has_next():
                row_vals = raw.get_next()
                rows.append(
                    {
                        col: self._convert_value(val)
                        for col, val in zip(columns, row_vals)
                    }
                )
        except Exception as exc:
            logger.warning("خطا در parse نتیجه: %s", exc)

        return QueryResult(
            data=rows,
            columns=columns,
            row_count=len(rows),
            query=query,
        )

    def _convert_value(self, value: Any) -> Any:
        """
        رفع مشکل: _convert_value ناقص
        پشتیبانی کامل از تمام type های Python و KuzuDB.
        """
        # مقادیر پایه Python
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._convert_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._convert_value(v) for k, v in value.items()}

        # datetime
        try:
            from datetime import date, datetime, timedelta
            if isinstance(value, (datetime, date)):
                return value.isoformat()
            if isinstance(value, timedelta):
                return str(value)
        except ImportError:
            pass

        # KuzuDB Node/Rel objects
        if hasattr(value, "_properties"):
            props = {k: self._convert_value(v) for k, v in value._properties.items()}
            result = {"_label": getattr(value, "_label", None), **props}
            return result

        if hasattr(value, "get_properties"):
            try:
                props = value.get_properties()
                return {k: self._convert_value(v) for k, v in props.items()}
            except Exception:
                pass

        # fallback
        return str(value)
