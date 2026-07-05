import praw
import feedparser
from newspaper import Article
from loguru import logger
from external_sources.data_collector.models import CollectedItem, CollectionResult, DataSource, DataDomain
from external_sources.data_collector.helper import DataHelper
from external_sources.data_collector.exceptions import AuthenticationError, CollectionError
import time
from external_sources.data_collector.timeout import with_timeout



class SocialCollector:
    """جمع‌آوری Reddit، RSS، اخبار"""

    def __init__(
        self,
        reddit_client_id:     str | None = None,
        reddit_client_secret: str | None = None,
        reddit_user_agent:    str = "DataCollector/1.0",
    ):
        self._reddit_creds = {
            "client_id":     reddit_client_id,
            "client_secret": reddit_client_secret,
            "user_agent":    reddit_user_agent,
        }
        self._reddit = None

    def _get_reddit(self) -> praw.Reddit:
        if self._reddit is None:
            if not self._reddit_creds["client_id"]:
                raise AuthenticationError("Reddit credentials not set")
            self._reddit = praw.Reddit(**self._reddit_creds)
        return self._reddit

    # ─── Reddit ─────────────────────────────────────────────────────────
    @with_timeout(30.0)
    def search_reddit(
        self,
        query: str,
        subreddit: str = "all",
        max_results: int = 20,
        sort: str = "relevance",    # relevance | hot | new | top
    ) -> CollectionResult:
        start = time.time()
        items, errors = [], []

        try:
            reddit = self._get_reddit()
            sub = reddit.subreddit(subreddit)
            posts = sub.search(query, limit=max_results, sort=sort)

            for post in posts:
                content = post.selftext or post.title
                if len(content) < 20:
                    continue

                items.append(CollectedItem(
                    id=DataHelper.make_id("reddit", post.id),
                    title=post.title,
                    content=DataHelper.clean_text(content),
                    summary=DataHelper.truncate(content, 300),
                    source=DataSource.REDDIT,
                    domain=DataDomain.SOCIAL,
                    url=f"https://reddit.com{post.permalink}",
                    author=str(post.author),
                    published_at=DataHelper.normalize_date(post.created_utc),
                    tags=[post.link_flair_text or "", post.subreddit.display_name],
                    metadata={
                        "score":        post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "num_comments": post.num_comments,
                        "subreddit":    post.subreddit.display_name,
                    }
                ))
        except AuthenticationError:
            raise
        except Exception as e:
            errors.append(f"Reddit: {e}")

        return CollectionResult(
            success=len(items) > 0,
            source=DataSource.REDDIT,
            total=len(items),
            items=items,
            errors=errors,
            elapsed_sec=time.time() - start,
            query=query
        )
        
    @with_timeout(20.0)
    def get_subreddit_hot(
        self,
        subreddit: str,
        max_results: int = 20
    ) -> CollectionResult:
        """پست‌های hot یک subreddit"""
        start = time.time()
        items, errors = [], []

        try:
            reddit = self._get_reddit()
            posts = reddit.subreddit(subreddit).hot(limit=max_results)

            for post in posts:
                items.append(CollectedItem(
                    id=DataHelper.make_id("reddit", post.id),
                    title=post.title,
                    content=DataHelper.clean_text(post.selftext or post.title),
                    source=DataSource.REDDIT,
                    domain=DataDomain.SOCIAL,
                    url=f"https://reddit.com{post.permalink}",
                    author=str(post.author),
                    published_at=DataHelper.normalize_date(post.created_utc),
                    metadata={"score": post.score, "subreddit": subreddit}
                ))
        except Exception as e:
            errors.append(f"Reddit hot: {e}")

        return CollectionResult(
            success=len(items) > 0,
            source=DataSource.REDDIT,
            total=len(items),
            items=items,
            errors=errors,
            elapsed_sec=time.time() - start
        )

    # ─── RSS Feeds ───────────────────────────────────────────────────────
    @with_timeout(3.0)#20.0
    def fetch_rss(
        self,
        feed_urls: list[str],
        max_per_feed: int = 10
    ) -> CollectionResult:
        start = time.time()
        items, errors = [], []

        for url in feed_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:max_per_feed]:
                    content = (
                        entry.get("summary") or
                        entry.get("description") or
                        entry.get("title", "")
                    )

                    items.append(CollectedItem(
                        id=DataHelper.make_id("rss", entry.get("id", url)),
                        title=entry.get("title", ""),
                        content=DataHelper.clean_text(content),
                        source=DataSource.RSS_FEED,
                        domain=DataDomain.NEWS,
                        url=entry.get("link", ""),
                        author=entry.get("author", ""),
                        published_at=DataHelper.normalize_date(
                            entry.get("published", entry.get("updated"))
                        ),
                        metadata={"feed_url": url}
                    ))
            except Exception as e:
                errors.append(f"RSS [{url}]: {e}")

        res = CollectionResult(
            success=len(items) > 0,
            source=DataSource.RSS_FEED,
            total=len(items),
            items=items,
            errors=errors,
            elapsed_sec=time.time() - start

        )
        # فقط خلاصه در INFO — دامپ کامل آیتم‌ها لاگ را غیرقابل خواندن می‌کرد
        logger.info(
            "📡 RSS fetched | feeds={} | items={} | errors={} | {:.1f}s",
            len(feed_urls), res.total, len(res.errors), res.elapsed_sec,
        )
        if res.errors:
            logger.warning("RSS errors: {}", res.errors)
        logger.debug("RSS titles: {}", [i.title for i in items])
        return res
    # ─── RSS منابع خبری آماده ───────────────────────────────────────────

    POLITICAL_RSS = [
        "https://feeds.bbci.co.uk/news/politics/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
        "https://thehill.com/rss/syndicator/19109/feed",
        "https://feeds.npr.org/1014/rss.xml",
        "https://www.theguardian.com/politics/rss",
    ]

    FINANCE_RSS = [
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.wsj.com/xml/rss/3_7031.xml",
        "https://feeds.ft.com/rss/home/uk",
    ]

    SCIENCE_RSS = [
        "https://www.nature.com/nature.rss",
        "https://feeds.newscientist.com/full-text-rss",
        "https://rss.sciencedaily.com/top/science.xml",
        "https://feeds.phys.org/all.xml",
    ]

    def fetch_political_news(self, max_per_feed: int = 5) -> CollectionResult:
        return self.fetch_rss(self.POLITICAL_RSS, max_per_feed)

    def fetch_financial_news(self, max_per_feed: int = 5) -> CollectionResult:
        return self.fetch_rss(self.FINANCE_RSS, max_per_feed)

    def fetch_science_news(self, max_per_feed: int = 5) -> CollectionResult:
        return self.fetch_rss(self.SCIENCE_RSS, max_per_feed)
