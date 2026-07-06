"""
ذخیره‌سازی دائمی سشن‌ها و trace فلوی اجرا.

هر کوئری (query، پاسخ، مسیر گره‌ها، زمان‌ها، confidence) یک خط JSON در
`data/sessions/flow_history.jsonl` می‌شود؛ در startup آخرین رکوردها
دوباره لود می‌شوند تا تاریخچه‌ی tab «Flow» بعد از restart هم بماند.
export() یک اجرا را به‌صورت JSON (و SVG در صورت ارسال) کنار می‌گذارد.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("data/sessions/flow_history.jsonl")
_EXPORT_DIR = Path("data/sessions/exports")


class FlowSessionStore:
    """append-only JSONL — thread-safe و مقاوم به خطوط خراب"""

    def __init__(self, path: Path | str = _DEFAULT_PATH):
        self._path = Path(path)
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock, open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.warning("Session store append failed: %s", e)

    def load_recent(self, limit: int = 200) -> list[dict]:
        """آخرین رکوردها (به همان ترتیب زمانی که ذخیره شده‌اند)"""
        if not self._path.exists():
            return []
        records: list[dict] = []
        try:
            with open(self._path, encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-limit:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if records:
                logger.info(
                    "💾 %d saved flow session(s) loaded from %s",
                    len(records), self._path,
                )
        except Exception as e:
            logger.warning("Session store load failed: %s", e)
        return records

    def export(
        self,
        record: dict,
        svg: Optional[str] = None,
        out_dir: Path | str = _EXPORT_DIR,
    ) -> list[str]:
        """ذخیره‌ی یک اجرا به‌صورت فایل JSON (+ SVG) — خروجی: مسیر فایل‌ها"""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe_q = "".join(
            c if c.isalnum() else "_" for c in str(record.get("query", ""))[:30]
        ).strip("_") or "flow"
        base = out / f"{stamp}_{safe_q}"

        paths: list[str] = []
        json_path = base.with_suffix(".json")
        json_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        paths.append(str(json_path))

        if svg:
            svg_path = base.with_suffix(".svg")
            svg_path.write_text(svg, encoding="utf-8")
            paths.append(str(svg_path))

        logger.info("💾 Flow session exported: %s", paths)
        return paths


_store: Optional[FlowSessionStore] = None
_store_lock = threading.Lock()


def get_session_store() -> FlowSessionStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = FlowSessionStore()
    return _store
