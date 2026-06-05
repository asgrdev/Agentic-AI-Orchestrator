import os
from pathlib import Path
import urllib.request
from typing import Any

# نسخه اصلاح شده بدون بررسی checksum
def patched_download(url: str, rootstr: str, in_memory: bool = False) -> bytes:
    """
    Download a file from a URL into a local folder.
    
    Modified to skip SHA256 check for local files.
    """
    root = Path(rootstr).expanduser()
    filename = os.path.basename(url)
    download_target = root / filename
    
    if download_target.exists():
        # فقط بررسی کن اگر فایل وجود دارد
        try:
            with open(download_target, "rb") as f:
                model_bytes = f.read()
            print(f"Using existing model file: {download_target}")
            return model_bytes
        except Exception as e:
            print(f"Error reading existing file: {e}")
            # ادامه بده تا دانلود کند
    
    # دانلود فایل
    root.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {url} to {download_target}")
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        while True:
            buffer = source.read(8192)
            if not buffer:
                break
            output.write(buffer)
    
    with open(download_target, "rb") as f:
        model_bytes = f.read()
    
    return model_bytes

def install_whisper_download_patch(whisper_module: Any) -> bool:
    """
    Patch whisper download hook without loading any model at import-time.
    Returns True when patch is installed/already installed.
    """
    try:
        if whisper_module is None:
            return False
        cur = getattr(whisper_module, "_download", None)
        if cur is patched_download:
            return True
        whisper_module._download = patched_download
        return True
    except Exception:
        return False
