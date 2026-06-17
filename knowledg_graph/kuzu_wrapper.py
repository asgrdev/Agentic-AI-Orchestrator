"""
KuzuDB Wrapper with Error Handling and Fallback Mechanism
=========================================================

این ماژول یک wrapper امن برای KuzuDB فراهم می‌کند که:
1. خطاهای import را مدیریت می‌کند
2. پیام‌های خطای واضح ارائه می‌دهد
3. fallback mechanism برای حالت‌های بدون kuzu دارد
4. راهنمای نصب ارائه می‌دهد

استفاده:
    from knowledg_graph.kuzu_wrapper import get_kuzu_client, KuzuAvailable
    
    if KuzuAvailable:
        client = get_kuzu_client(db_path="./data/kuzu_db")
    else:
        # Use fallback or raise error
        raise RuntimeError("KuzuDB is required but not installed")
"""

import logging
import sys
from typing import Optional, Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Check KuzuDB Availability
# ─────────────────────────────────────────────

KuzuAvailable = False
KuzuImportError: Optional[Exception] = None

try:
    import kuzu
    KuzuAvailable = True
    logger.info(f"KuzuDB successfully imported (version: {kuzu.__version__})")
except ImportError as e:
    KuzuImportError = e
    logger.warning(f"KuzuDB not available: {e}")
except Exception as e:
    KuzuImportError = e
    logger.error(f"Unexpected error importing KuzuDB: {e}")


# ─────────────────────────────────────────────
# Installation Instructions
# ─────────────────────────────────────────────

INSTALLATION_GUIDE = """
╔══════════════════════════════════════════════════════════════╗
║                   KuzuDB Installation Guide                   ║
╚══════════════════════════════════════════════════════════════╝

KuzuDB is required for graph database functionality.

📦 Installation:
   pip install kuzu

🔧 For specific versions:
   pip install kuzu==0.5.0

🐍 Python Requirements:
   - Python 3.8+
   - pip or conda

📚 Documentation:
   https://kuzudb.com/docs/

⚠️  Platform Notes:
   - macOS: Requires macOS 10.15+ (Catalina or later)
   - Linux: Requires glibc 2.17+
   - Windows: Requires Windows 10+

🔍 Verify Installation:
   python3 -c "import kuzu; print(kuzu.__version__)"

💡 Troubleshooting:
   - Update pip: pip install --upgrade pip
   - Use virtual environment: python3 -m venv venv
   - Check Python version: python3 --version
   - Try with sudo (Linux): sudo pip3 install kuzu

For more help, visit: https://github.com/kuzudb/kuzu
"""


# ─────────────────────────────────────────────
# Error Classes
# ─────────────────────────────────────────────

class KuzuNotInstalledError(ImportError):
    """Raised when KuzuDB is not installed."""
    
    def __init__(self, original_error: Optional[Exception] = None):
        self.original_error = original_error
        message = (
            "KuzuDB is not installed or failed to import.\n"
            f"{INSTALLATION_GUIDE}"
        )
        if original_error:
            message += f"\n\nOriginal error: {original_error}"
        super().__init__(message)


class KuzuConnectionError(Exception):
    """Raised when connection to KuzuDB fails."""
    pass


# ─────────────────────────────────────────────
# Wrapper Functions
# ─────────────────────────────────────────────

def ensure_kuzu_available() -> None:
    """
    بررسی می‌کند که KuzuDB در دسترس است.
    در غیر این صورت، خطای واضح با راهنمای نصب می‌دهد.
    
    Raises:
        KuzuNotInstalledError: اگر KuzuDB نصب نباشد
    """
    if not KuzuAvailable:
        raise KuzuNotInstalledError(KuzuImportError)


def get_kuzu_client(db_path: str, **kwargs) -> Any:
    """
    یک KuzuClient ایجاد می‌کند با error handling کامل.
    
    Args:
        db_path: مسیر دیتابیس
        **kwargs: پارامترهای اضافی برای KuzuClient
    
    Returns:
        KuzuClient instance
    
    Raises:
        KuzuNotInstalledError: اگر KuzuDB نصب نباشد
        KuzuConnectionError: اگر اتصال ناموفق باشد
    """
    ensure_kuzu_available()
    
    try:
        # Try to import from project structure
        try:
            from knowledg_graph.kuzudb_package.client import KuzuClient
        except ImportError:
            # Fallback: use direct kuzu
            import kuzu
            from pathlib import Path
            
            # Kuzu 0.11+ expects directory path, not file
            db_dir = Path(db_path)
            db_dir.mkdir(parents=True, exist_ok=True)
            
            db = kuzu.Database(str(db_dir))
            conn = kuzu.Connection(db)
            logger.info(f"KuzuDB connected directly: {db_path}")
            return conn
        
        client = KuzuClient(db_path, **kwargs)
        client.connect()
        logger.info(f"KuzuDB client connected successfully: {db_path}")
        return client
    except Exception as e:
        logger.error(f"Failed to create KuzuDB client: {e}")
        raise KuzuConnectionError(f"Failed to connect to KuzuDB at {db_path}: {e}")


def get_async_kuzu_manager(db_path: str, **kwargs) -> Any:
    """
    یک AsyncKuzuManager ایجاد می‌کند با error handling کامل.
    
    Args:
        db_path: مسیر دیتابیس
        **kwargs: پارامترهای اضافی
    
    Returns:
        AsyncKuzuManager instance
    
    Raises:
        KuzuNotInstalledError: اگر KuzuDB نصب نباشد
    """
    ensure_kuzu_available()
    
    try:
        from knowledg_graph.kuzudb_package.async_manager import AsyncKuzuManager
        manager = AsyncKuzuManager(db_path, **kwargs)
        logger.info(f"AsyncKuzuManager created successfully: {db_path}")
        return manager
    except Exception as e:
        logger.error(f"Failed to create AsyncKuzuManager: {e}")
        raise KuzuConnectionError(f"Failed to create AsyncKuzuManager: {e}")


# ─────────────────────────────────────────────
# Fallback Mechanism
# ─────────────────────────────────────────────

class MockKuzuClient:
    """
    یک mock client برای تست‌ها یا حالت‌های بدون KuzuDB.
    تمام عملیات را log می‌کند اما هیچ کاری انجام نمی‌دهد.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        logger.warning(f"Using MockKuzuClient (KuzuDB not available): {db_path}")
    
    def connect(self):
        logger.debug("MockKuzuClient.connect() called")
        return self
    
    def disconnect(self):
        logger.debug("MockKuzuClient.disconnect() called")
    
    def execute(self, query: str, params: Optional[dict] = None):
        logger.debug(f"MockKuzuClient.execute() called: {query[:100]}...")
        return MockQueryResult()
    
    def __enter__(self):
        return self.connect()
    
    def __exit__(self, *args):
        self.disconnect()


class MockQueryResult:
    """Mock result برای MockKuzuClient."""
    
    def __init__(self):
        self.data = []
        self.columns = []
        self.row_count = 0
    
    def first(self):
        return None
    
    def scalar(self, key: str, default=None):
        return default
    
    def __iter__(self):
        return iter([])
    
    def __len__(self):
        return 0
    
    def __bool__(self):
        return False


def get_kuzu_client_or_mock(db_path: str, allow_mock: bool = False, **kwargs) -> Any:
    """
    KuzuClient یا MockKuzuClient برمی‌گرداند.
    
    Args:
        db_path: مسیر دیتابیس
        allow_mock: اگر True باشد، در صورت نبود KuzuDB از mock استفاده می‌کند
        **kwargs: پارامترهای اضافی
    
    Returns:
        KuzuClient یا MockKuzuClient
    
    Raises:
        KuzuNotInstalledError: اگر KuzuDB نصب نباشد و allow_mock=False
    """
    if KuzuAvailable:
        return get_kuzu_client(db_path, **kwargs)
    elif allow_mock:
        logger.warning("KuzuDB not available, using MockKuzuClient")
        return MockKuzuClient(db_path)
    else:
        raise KuzuNotInstalledError(KuzuImportError)


# ─────────────────────────────────────────────
# Diagnostic Functions
# ─────────────────────────────────────────────

def print_kuzu_status():
    """وضعیت KuzuDB را چاپ می‌کند."""
    print("\n" + "="*60)
    print("KuzuDB Status Check")
    print("="*60)
    
    if KuzuAvailable:
        print("✅ KuzuDB is AVAILABLE")
        try:
            import kuzu
            print(f"   Version: {kuzu.__version__}")
            print(f"   Location: {kuzu.__file__}")
        except Exception as e:
            print(f"   ⚠️  Error getting details: {e}")
    else:
        print("❌ KuzuDB is NOT AVAILABLE")
        if KuzuImportError:
            print(f"   Error: {KuzuImportError}")
        print("\n" + INSTALLATION_GUIDE)
    
    print("="*60 + "\n")


def get_kuzu_info() -> dict:
    """اطلاعات KuzuDB را به صورت dict برمی‌گرداند."""
    info = {
        "available": KuzuAvailable,
        "error": str(KuzuImportError) if KuzuImportError else None,
    }
    
    if KuzuAvailable:
        try:
            import kuzu
            info["version"] = kuzu.__version__
            info["location"] = kuzu.__file__
        except Exception as e:
            info["details_error"] = str(e)
    
    return info


# ─────────────────────────────────────────────
# CLI for Testing
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print_kuzu_status()
    
    # Test connection if available
    if KuzuAvailable:
        print("\n🧪 Testing KuzuDB connection...")
        try:
            client = get_kuzu_client("./test_kuzu_db")
            print("✅ Connection successful!")
            client.disconnect()
        except Exception as e:
            print(f"❌ Connection failed: {e}")
    
    sys.exit(0 if KuzuAvailable else 1)

