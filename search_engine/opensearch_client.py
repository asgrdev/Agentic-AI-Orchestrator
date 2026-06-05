from opensearchpy import AsyncOpenSearch


class OpenSearchEngine:
    """
    OpenSearch برای:
    - BM25 text search (دقت بالا در کلمات کلیدی)
    - Aggregation و analytics
    - Document-level indexing
    - فیلتر بر اساس metadata
    """

    def __init__(self, config: dict):
        self._client = AsyncOpenSearch(
            hosts=config["opensearch"]["hosts"],
            http_auth=(
                config["opensearch"]["user"],
                config["opensearch"]["password"],
            ),
            use_ssl=config["opensearch"].get("use_ssl", False),
        )
        self.index = config["opensearch"].get("index", "knowledge_docs")

    async def index_document(self, doc: dict):
        await self._client.index(
            index=self.index,
            id=doc["id"],
            body={
                "content":    doc["content"],
                "title":      doc.get("title", ""),
                "source":     doc.get("source", ""),
                "entity_ids": doc.get("entity_ids", []),
                "timestamp":  doc.get("timestamp"),
                "metadata":   doc.get("metadata", {}),
            },
        )

    async def bm25_search(
        self,
        query:    str,
        filters:  dict | None = None,
        top_k:    int = 10,
    ) -> list[dict]:
        must_clauses = [
            {
                "multi_match": {
                    "query":  query,
                    "fields": ["content^2", "title^3"],
                    "type":   "best_fields",
                }
            }
        ]

        # اضافه کردن فیلتر اختیاری
        filter_clauses = []
        if filters:
            for key, val in filters.items():
                filter_clauses.append({"term": {key: val}})

        body = {
            "query": {
                "bool": {
                    "must":   must_clauses,
                    "filter": filter_clauses,
                }
            },
            "size": top_k,
            "_source": ["content", "title", "source", "entity_ids"],
            "highlight": {
                "fields": {"content": {}}
            },
        }

        resp = await self._client.search(index=self.index, body=body)
        return [
            {
                "id":         hit["_id"],
                "content":    hit["_source"]["content"],
                "source":     hit["_source"]["source"],
                "score":      hit["_score"],
                "entity_ids": hit["_source"].get("entity_ids", []),
                "highlights": hit.get("highlight", {}).get("content", []),
            }
            for hit in resp["hits"]["hits"]
        ]
