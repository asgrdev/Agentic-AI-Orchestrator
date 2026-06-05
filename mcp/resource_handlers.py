# resource_handlers.py

import json
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def resource_last_collection():

    file = CACHE_DIR / "collected_items.json"

    if not file.exists():
        return {"items": []}

    with open(file, "r") as f:
        data = json.load(f)

    return data


def resource_rag_corpus():

    file = CACHE_DIR / "rag_corpus.json"

    if not file.exists():
        return {"chunks": []}

    with open(file, "r") as f:
        data = json.load(f)

    return data
