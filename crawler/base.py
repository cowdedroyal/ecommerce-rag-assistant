"""
Base crawler with anti-detection, rate limiting, retry logic, and async Playwright support.
"""

import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

logger = logging.getLogger(__name__)

# Rotating User-Agent pool covering major browsers on different OS
USER_AGENTS: List[str] = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Firefox on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    # Safari on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    # Chrome on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


@dataclass
class CrawlStats:
    """Tracks crawl session statistics."""

    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    retried: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful / self.total_requests

    def summary(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful": self.successful,
            "failed": self.failed,
            "retried": self.retried,
            "elapsed_seconds": round(self.elapsed, 1),
            "success_rate": round(self.success_rate, 4),
        }


class RateLimiter:
    """Token-bucket rate limiter scoped to requests per minute."""

    def __init__(self, max_per_minute: int = 20):
        self._max = max_per_minute
        self._timestamps: List[float] = []

    async def acquire(self) -> None:
        """Block until a request slot is available."""
        now = time.time()
        # Purge timestamps older than 60 s
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self._max:
            wait = 60 - (now - self._timestamps[0])
            if wait > 0:
                logger.debug("Rate limit reached, waiting %.1f s", wait)
                await asyncio.sleep(wait)
        self._timestamps.append(time.time())


class BaseCrawler(ABC):
    """
    Abstract base for all e-commerce crawlers.

    Responsibilities handled here:
      - Browser lifecycle (Playwright async Chromium)
      - Anti-detection (random UA, stealth JS, viewport jitter)
      - Rate limiting (requests per minute)
      - Random delay between requests
      - Automatic retry with exponential back-off
      - Structured logging and stats collection
      - JSON persistence of raw crawl results

    Subclasses implement ``crawl()`` which orchestrates category iteration
    and calls ``_fetch_page`` / ``_extract_products`` as needed.
    """

    def __init__(
        self,
        output_path: Path,
        categories: Optional[List[str]] = None,
        max_per_category: int = 100,
        delay_min: float = 2.0,
        delay_max: float = 5.0,
        max_retries: int = 3,
        requests_per_minute: int = 20,
        headless: bool = True,
    ):
        self.output_path = Path(output_path)
        self.categories = categories or []
        self.max_per_category = max_per_category
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.max_retries = max_retries
        self.headless = headless

        self._rate_limiter = RateLimiter(requests_per_minute)
        self._stats = CrawlStats()
        self._products: List[Dict[str, Any]] = []

        # Playwright handles (initialised in ``_init_browser``)
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    # ------------------------------------------------------------------
    # Browser lifecycle
    # ------------------------------------------------------------------

    async def _init_browser(self) -> None:
        """Launch a Chromium instance with anti-detection tweaks."""
        self._playwright = await async_playwright().start()

        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        ua = random.choice(USER_AGENTS)
        viewport = {
            "width": random.randint(1280, 1920),
            "height": random.randint(800, 1080),
        }

        self._context = await self._browser.new_context(
            user_agent=ua,
            viewport=viewport,
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
        )

        # Inject stealth script to mask webdriver flag
        await self._context.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN', 'zh', 'en']});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            window.chrome = { runtime: {} };
            """
        )

        logger.info(
            "Browser launched (headless=%s, UA=%s, viewport=%s)",
            self.headless,
            ua[:60],
            viewport,
        )

    async def _close_browser(self) -> None:
        """Gracefully close browser and Playwright."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser closed")

    # ------------------------------------------------------------------
    # Page helpers
    # ------------------------------------------------------------------

    async def _new_page(self) -> Page:
        """Open a new tab from the current browser context."""
        if self._context is None:
            raise RuntimeError("Browser not initialised; call _init_browser first")
        return await self._context.new_page()

    async def _random_delay(self) -> None:
        """Sleep a random interval between configured bounds."""
        delay = random.uniform(self.delay_min, self.delay_max)
        logger.debug("Sleeping %.2f s", delay)
        await asyncio.sleep(delay)

    async def _scroll_page(self, page: Page, scrolls: int = 3) -> None:
        """Simulate human-like scrolling to trigger lazy-loaded content."""
        for _ in range(scrolls):
            await page.evaluate(
                "window.scrollBy(0, window.innerHeight * (0.6 + Math.random() * 0.4))"
            )
            await asyncio.sleep(random.uniform(0.5, 1.5))

    # ------------------------------------------------------------------
    # Fetch with retry
    # ------------------------------------------------------------------

    async def _fetch_page(self, page: Page, url: str) -> bool:
        """
        Navigate to *url* with rate limiting and retry logic.

        Returns ``True`` on success, ``False`` after all retries exhausted.
        """
        for attempt in range(1, self.max_retries + 1):
            await self._rate_limiter.acquire()
            self._stats.total_requests += 1

            try:
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                if resp and resp.ok:
                    self._stats.successful += 1
                    await self._random_delay()
                    return True

                status = resp.status if resp else "no response"
                logger.warning(
                    "Non-OK status %s for %s (attempt %d/%d)",
                    status,
                    url,
                    attempt,
                    self.max_retries,
                )
            except Exception as exc:
                logger.warning(
                    "Request error for %s (attempt %d/%d): %s",
                    url,
                    attempt,
                    self.max_retries,
                    exc,
                )

            self._stats.retried += 1

            if attempt < self.max_retries:
                backoff = (2 ** attempt) + random.uniform(0, 1)
                logger.info("Retrying in %.1f s ...", backoff)
                await asyncio.sleep(backoff)

        self._stats.failed += 1
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_products(self) -> None:
        """Write collected products to JSON, creating parent dirs as needed."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as fh:
            json.dump(self._products, fh, ensure_ascii=False, indent=2)
        logger.info("Saved %d products to %s", len(self._products), self.output_path)

    def _append_product(self, product: Dict[str, Any]) -> None:
        """Append a single validated product record."""
        if not product.get("title"):
            return
        self._products.append(product)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def crawl(self) -> List[Dict[str, Any]]:
        """
        Run the full crawl pipeline.

        Subclasses must:
          1. Call ``_init_browser``
          2. Iterate categories and pages
          3. Call ``_fetch_page`` / ``_extract_products``
          4. Call ``_save_products``
          5. Call ``_close_browser`` in a finally block

        Returns the list of collected product dicts.
        """

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats.summary()
