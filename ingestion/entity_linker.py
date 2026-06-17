import httpx
import wikipediaapi
from ingestion.datasets.entities import Entity


class EntityLinker:

    def __init__(self, config: dict):

        self.wikidata_url = "https://www.wikidata.org/w/api.php"

        self.client = httpx.AsyncClient(timeout=10)

        self.cache = {}

        self.wiki = wikipediaapi.Wikipedia(    user_agent='MyAwesomeApp/1.0 (contact@myemail.com)',language="en")

    async def link_entities(self, entities: list[Entity]):

        results = []

        for e in entities:

            if e.normalized in self.cache:

                cached = self.cache[e.normalized]

                e.linked_id = cached["id"]
                e.kb_url = cached["url"]
                e.description = cached["desc"]
                e.wikipedia_url = cached["wiki"]

                results.append(e)
                continue

            result = await self._search_wikidata(e.text)

            if result:

                e.linked_id = result["id"]
                e.kb_url = result["url"]

                wiki_data = self._get_wikipedia(e.text)

                if wiki_data:

                    e.description = wiki_data["summary"]
                    e.wikipedia_url = wiki_data["url"]

                self.cache[e.normalized] = {
                    "id": e.linked_id,
                    "url": e.kb_url,
                    "desc": e.description,
                    "wiki": e.wikipedia_url
                }

            results.append(e)

        return results

    async def _search_wikidata(self, text):

        try:

            r = await self.client.get(
                self.wikidata_url,
                params={
                    "action": "wbsearchentities",
                    "search": text,
                    "language": "en",
                    "format": "json",
                    "limit": 1
                }
            )

            data = r.json()

            if not data["search"]:
                return None

            item = data["search"][0]

            return {
                "id": item["id"],
                "url": item["concepturi"]
            }

        except Exception:
            return None

    def _get_wikipedia(self, text):

        page = self.wiki.page(text)

        if not page.exists():
            return None

        return {
            "summary": page.summary[:500],
            "url": page.fullurl
        }

    async def _search_kb(
        self, text: str, entity_type: str
    ) -> dict | None:
        # اول KB داخلی
        if self.internal_kb_url:
            result = await self._search_internal(text, entity_type)
            if result:
                return result
        # بعد WikiData
        return await self._search_wikidata(text)

    async def _search_wikidata(self, text: str) -> dict | None:
        try:
            resp = await self._client.get(
                self.wikidata_url,
                params={
                    "action": "wbsearchentities",
                    "search": text,
                    "language": "en",
                    "format": "json",
                    "limit": 1,
                },
            )
            data = resp.json()
            if data.get("search"):
                item = data["search"][0]
                return {
                    "id":  item["id"],
                    "url": item["concepturi"],
                }
        except Exception:
            pass
        return None

    async def _search_internal(
        self, text: str, entity_type: str
    ) -> dict | None:
        # جستجو در KB داخلی (Neo4j یا جدول مجزا)
        return None   # پیاده‌سازی بر اساس KB داخلی
