import arxiv
from pyalex import Works
from semanticscholar import SemanticScholar
from loguru import logger
from external_sources.data_collector.models import CollectedItem, CollectionResult, DataSource, DataDomain
from external_sources.data_collector.helper import DataHelper
from external_sources.data_collector.exceptions import CollectionError
import time
from external_sources.data_collector.timeout import with_timeout



class ScientificCollector:
    """جمع‌آوری ArXiv، OpenAlex، SemanticScholar"""

    def __init__(self):
        self.ss = SemanticScholar()
        self.arxiv_client = arxiv.Client()

    # ─── ArXiv ──────────────────────────────────────────────────────────
    @with_timeout(30.0)
    def search_arxiv(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance"
    ) -> CollectionResult:
        start = time.time()
        items, errors = [], []

        sort_map = {
            "relevance":  arxiv.SortCriterion.Relevance,
            "date":       arxiv.SortCriterion.SubmittedDate,
            "citations":  arxiv.SortCriterion.Relevance,
        }

        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_map.get(sort_by, arxiv.SortCriterion.Relevance)
            )

            for paper in self.arxiv_client.results(search):
                items.append(CollectedItem(
                    id=DataHelper.make_id("arxiv", paper.get_short_id()),
                    title=paper.title,
                    content=DataHelper.clean_text(paper.summary),
                    summary=DataHelper.truncate(paper.summary, 300),
                    source=DataSource.ARXIV,
                    domain=DataDomain.SCIENTIFIC,
                    url=paper.pdf_url,
                    author=", ".join(a.name for a in paper.authors[:3]),
                    published_at=DataHelper.normalize_date(paper.published),
                    tags=paper.categories,
                    metadata={
                        "doi": paper.doi,
                        "categories": paper.categories,
                        "arxiv_id": paper.get_short_id(),
                    }
                ))
        except Exception as e:
            errors.append(f"ArXiv: {e}")

        return CollectionResult(
            success=len(items) > 0,
            source=DataSource.ARXIV,
            total=len(items),
            items=items,
            errors=errors,
            elapsed_sec=time.time() - start,
            query=query
        )

    # ─── OpenAlex ───────────────────────────────────────────────────────
    @with_timeout(30.0)
    def search_openalex(
        self,
        query: str,
        max_results: int = 10,
    ) -> CollectionResult:
        start = time.time()
        items, errors = [], []

        try:
            works = (
                Works()
                .search(query)
                .filter(has_abstract=True)
                .sort(cited_by_count="desc")
                .get(per_page=max_results)
            )

            for work in works:
                abstract = work.get("abstract") or ""
                if not abstract:
                    continue

                authors = work.get("authorships", [])
                author_names = [
                    a["author"]["display_name"]
                    for a in authors[:3]
                    if a.get("author")
                ]

                items.append(CollectedItem(
                    id=DataHelper.make_id("openalex", work["id"]),
                    title=work.get("title", ""),
                    content=DataHelper.clean_text(abstract),
                    source=DataSource.OPENALEX,
                    domain=DataDomain.SCIENTIFIC,
                    url=work.get("doi", ""),
                    author=", ".join(author_names),
                    published_at=DataHelper.normalize_date(work.get("publication_date")),
                    metadata={
                        "citations": work.get("cited_by_count", 0),
                        "open_access": DataHelper.safe_get(work, "open_access", "is_oa"),
                    }
                ))
        except Exception as e:
            errors.append(f"OpenAlex: {e}")

        return CollectionResult(
            success=len(items) > 0,
            source=DataSource.OPENALEX,
            total=len(items),
            items=items,
            errors=errors,
            elapsed_sec=time.time() - start,
            query=query
        )

    # ─── Semantic Scholar ────────────────────────────────────────────────
    @with_timeout(30.0)
    def search_semantic_scholar(
        self,
        query: str,
        max_results: int = 10,
    ) -> CollectionResult:
        start = time.time()
        items, errors = [], []

        try:
            results = self.ss.search_paper(
                query,
                limit=max_results,
                fields=["title", "abstract", "authors", "year",
                        "citationCount", "externalIds", "url"]
            )

            for paper in results:
                if not paper.abstract:
                    continue

                items.append(CollectedItem(
                    id=DataHelper.make_id("ss", paper.paperId or query),
                    title=paper.title or "",
                    content=DataHelper.clean_text(paper.abstract),
                    source=DataSource.SEMANTIC_SCHOLAR,
                    domain=DataDomain.SCIENTIFIC,
                    url=paper.url or "",
                    author=", ".join(
                        a["name"] for a in (paper.authors or [])[:3]
                    ),
                    published_at=DataHelper.normalize_date(
                        f"{paper.year}-01-01" if paper.year else None
                    ),
                    metadata={
                        "citations": paper.citationCount or 0,
                        "paper_id": paper.paperId,
                    }
                ))
        except Exception as e:
            errors.append(f"SemanticScholar: {e}")

        return CollectionResult(
            success=len(items) > 0,
            source=DataSource.SEMANTIC_SCHOLAR,
            total=len(items),
            items=items,
            errors=errors,
            elapsed_sec=time.time() - start,
            query=query
        )
