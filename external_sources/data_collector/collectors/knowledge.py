import wikipediaapi
from datasets import load_dataset
from loguru import logger
from external_sources.data_collector.models import CollectedItem, CollectionResult, DataSource, DataDomain
from external_sources.data_collector.helper import DataHelper
from external_sources.data_collector.exceptions import CollectionError
import time
from external_sources.data_collector.timeout import with_timeout

class KnowledgeCollector:
    """جمع‌آوری Wikipedia و HuggingFace Datasets"""

    def __init__(self, language: str = "en"):
        self.language = language
        self.wiki = wikipediaapi.Wikipedia(
            user_agent="DataCollector/1.0",
            language=language
        )

    # ─── Wikipedia ──────────────────────────────────────────────────────

    @DataHelper.retry(max_attempts=3, delay=1.0)
    def get_wikipedia_page(self, title: str) -> CollectedItem | None:
        page = self.wiki.page(title)
        if not page.exists():
            return None

        return CollectedItem(
            id=DataHelper.make_id("wikipedia", title),
            title=page.title,
            content=DataHelper.clean_text(page.text),
            summary=DataHelper.truncate(page.summary, 500),
            source=DataSource.WIKIPEDIA,
            domain=DataDomain.KNOWLEDGE,
            url=page.fullurl,
            language=self.language,
            tags=DataHelper.extract_keywords(page.text),
            metadata={
                "sections": [s.title for s in page.sections],
                "links_count": len(page.links),
            }
        )
    @with_timeout(5.0)    
    def search_wikipedia(
        self,
        query: str,
        max_results: int = 10
    ) -> CollectionResult:
        start = time.time()
        items, errors = [], []

        try:
            search_results = self.wiki.search(query, results=max_results)
            for title in search_results:
                try:
                    item = self.get_wikipedia_page(title)
                    if item:
                        items.append(item)
                except Exception as e:
                    errors.append(f"Wikipedia [{title}]: {e}")
        except Exception as e:
            errors.append(f"Wikipedia search: {e}")

        return CollectionResult(
            success=len(items) > 0,
            source=DataSource.WIKIPEDIA,
            total=len(items),
            items=items,
            errors=errors,
            elapsed_sec=time.time() - start,
            query=query
        )

    # ─── HuggingFace Datasets ────────────────────────────────────────────
    @with_timeout(5.0)#120.0)
    def load_hf_dataset(
        self,
        dataset_name: str,
        subset: str | None = None,
        split: str = "train",
        max_rows: int = 100,
        text_column: str = "text",
        title_column: str | None = "title",
    ) -> CollectionResult:
        start = time.time()
        items, errors = [], []

        try:
            ds = load_dataset(
                dataset_name,
                subset,
                split=split,
                streaming=True   # streaming تا RAM نخوره
            )

            for i, row in enumerate(ds):
                if i >= max_rows:
                    break
                try:
                    content = row.get(text_column, "")
                    if not content:
                        continue

                    items.append(CollectedItem(
                        id=DataHelper.make_id(dataset_name, str(i)),
                        title=row.get(title_column, f"Row {i}") if title_column else f"Row {i}",
                        content=DataHelper.clean_text(str(content)),
                        source=DataSource.HUGGINGFACE,
                        domain=DataDomain.KNOWLEDGE,
                        language=DataHelper.detect_language(str(content)),
                        metadata={"dataset": dataset_name, "row": i}
                    ))
                except Exception as e:
                    errors.append(f"Row {i}: {e}")

        except Exception as e:
            errors.append(f"HuggingFace load: {e}")

        return CollectionResult(
            success=len(items) > 0,
            source=DataSource.HUGGINGFACE,
            total=len(items),
            items=items,
            errors=errors,
            elapsed_sec=time.time() - start,
        )
