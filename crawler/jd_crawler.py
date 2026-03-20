"""
JD.com (jd.com) product crawler.

Navigates JD search pages for each configured category, extracts product
listings, then visits individual product detail pages to collect full
metadata (specs, features, description).

Output: data/crawled/jd_products.json
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from playwright.async_api import Page

from crawler.base import BaseCrawler

logger = logging.getLogger(__name__)

# JD search URL template — keyword is URL-encoded
_SEARCH_URL = "https://search.jd.com/Search?keyword={keyword}&page={page}"

# Maximum search result pages to scan per category (JD returns 30 items/page)
_MAX_PAGES_PER_CATEGORY = 4


class JDCrawler(BaseCrawler):
    """
    Concrete crawler targeting JD.com.

    Strategy
    --------
    1. For each category keyword, iterate search result pages.
    2. On each search page, extract product card links.
    3. For each product link, visit the detail page and scrape structured data.
    4. Persist all results as JSON.

    The crawler automatically stops when ``max_per_category`` products have
    been collected for a given category or when pages are exhausted.
    """

    SOURCE = "jd"

    def __init__(
        self,
        output_path: Optional[Path] = None,
        categories: Optional[List[str]] = None,
        max_per_category: int = 100,
        **kwargs,
    ):
        output_path = output_path or Path("data/crawled/jd_products.json")
        super().__init__(
            output_path=output_path,
            categories=categories or [],
            max_per_category=max_per_category,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Search page parsing
    # ------------------------------------------------------------------

    async def _extract_product_links(self, page: Page) -> List[str]:
        """
        Return product detail URLs found on a JD search result page.

        JD renders product cards inside ``#J_goodsList .gl-item`` nodes;
        each card contains an ``<a>`` whose href points to the item page.
        """
        await self._scroll_page(page, scrolls=5)

        links: List[str] = []
        try:
            items = await page.query_selector_all("#J_goodsList .gl-item")
            for item in items:
                anchor = await item.query_selector(".p-img a")
                if anchor is None:
                    continue
                href = await anchor.get_attribute("href")
                if href:
                    url = href if href.startswith("http") else f"https:{href}"
                    links.append(url)
        except Exception as exc:
            logger.warning("Failed to extract product links: %s", exc)

        logger.debug("Found %d product links on page", len(links))
        return links

    # ------------------------------------------------------------------
    # Detail page parsing
    # ------------------------------------------------------------------

    async def _extract_product_detail(
        self, page: Page, category: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract structured product data from a JD item detail page.

        Fields: product_id, title, brand, category, price, rating,
        review_count, description, specs, features, source.
        """
        product: Dict[str, Any] = {"source": self.SOURCE, "category": category}

        try:
            # --- product_id from URL ---
            url = page.url
            match = re.search(r"/(\d+)\.html", url)
            product["product_id"] = match.group(1) if match else url

            # --- title ---
            title_el = await page.query_selector(".sku-name")
            product["title"] = (
                (await title_el.inner_text()).strip() if title_el else ""
            )

            # --- brand ---
            brand_el = await page.query_selector("#parameter-brand li")
            if brand_el:
                brand_text = await brand_el.inner_text()
                product["brand"] = brand_text.replace("品牌：", "").strip()
            else:
                product["brand"] = self._infer_brand(product.get("title", ""))

            # --- price ---
            product["price"] = await self._extract_price(page)

            # --- rating & review_count ---
            rating, review_count = await self._extract_rating(page)
            product["rating"] = rating
            product["review_count"] = review_count

            # --- description ---
            product["description"] = await self._extract_description(page)

            # --- specs ---
            product["specs"] = await self._extract_specs(page)

            # --- features ---
            product["features"] = await self._extract_features(page)

        except Exception as exc:
            logger.error("Error extracting product detail: %s", exc)
            return None

        # Validate minimum required fields
        if not product.get("title"):
            return None

        return product

    # ------------------------------------------------------------------
    # Field-level extractors
    # ------------------------------------------------------------------

    async def _extract_price(self, page: Page) -> Optional[float]:
        """Attempt to read the item price from multiple selectors."""
        selectors = [".p-price .price", ".summary-price .price", "span.price"]
        for sel in selectors:
            el = await page.query_selector(sel)
            if el:
                text = await el.inner_text()
                cleaned = re.sub(r"[^\d.]", "", text)
                try:
                    return float(cleaned)
                except ValueError:
                    continue
        return None

    async def _extract_rating(self, page: Page) -> tuple:
        """
        Return (rating: float | None, review_count: int | None).

        JD shows a "good-rate" percentage and comment count on detail pages.
        """
        rating = None
        review_count = None

        try:
            # Good-rate percentage (e.g. "97%好评") → scale to 0-5
            rate_el = await page.query_selector(".percent-con")
            if rate_el:
                rate_text = await rate_el.inner_text()
                pct_match = re.search(r"(\d+)", rate_text)
                if pct_match:
                    rating = round(float(pct_match.group(1)) / 100 * 5, 1)

            # Review count
            count_el = await page.query_selector("#comment-count a")
            if count_el:
                count_text = await count_el.inner_text()
                count_text = count_text.replace("+", "").replace("万", "0000")
                digits = re.sub(r"[^\d]", "", count_text)
                if digits:
                    review_count = int(digits)
        except Exception as exc:
            logger.debug("Rating extraction issue: %s", exc)

        return rating, review_count

    async def _extract_description(self, page: Page) -> str:
        """Pull the promotional copy shown below the title area."""
        selectors = [
            ".p-ad",
            ".ad-title",
            ".sku-name + div",
        ]
        parts: List[str] = []
        for sel in selectors:
            elements = await page.query_selector_all(sel)
            for el in elements:
                text = (await el.inner_text()).strip()
                if text:
                    parts.append(text)
        return " ".join(parts)[:500] if parts else ""

    async def _extract_specs(self, page: Page) -> Dict[str, str]:
        """
        Parse the specification table (规格参数) into a dict.

        JD nests spec rows inside ``#detail .Ptable .Ptable-item``.
        """
        specs: Dict[str, str] = {}
        try:
            rows = await page.query_selector_all(
                ".Ptable .Ptable-item dl.clearfix"
            )
            for row in rows:
                dt = await row.query_selector("dt")
                dd = await row.query_selector("dd")
                if dt and dd:
                    key = (await dt.inner_text()).strip()
                    val = (await dd.inner_text()).strip()
                    if key and val:
                        specs[key] = val

            # Fallback: parameter list
            if not specs:
                params = await page.query_selector_all(
                    "#detail .parameter2 li"
                )
                for li in params:
                    text = (await li.inner_text()).strip()
                    if "：" in text:
                        k, v = text.split("：", 1)
                        specs[k.strip()] = v.strip()
        except Exception as exc:
            logger.debug("Spec extraction issue: %s", exc)

        return specs

    async def _extract_features(self, page: Page) -> List[str]:
        """Collect selling-point bullet items."""
        features: List[str] = []
        try:
            items = await page.query_selector_all(".p-parameter li, .sui-tag span")
            for item in items:
                text = (await item.inner_text()).strip()
                if text and len(text) > 1:
                    features.append(text)
        except Exception as exc:
            logger.debug("Feature extraction issue: %s", exc)

        return features[:20]  # cap to avoid noise

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_brand(title: str) -> str:
        """Best-effort brand extraction from the product title."""
        known_brands = [
            "华为", "小米", "Apple", "苹果", "OPPO", "vivo", "三星",
            "联想", "戴尔", "惠普", "索尼", "佳能", "尼康", "飞利浦",
            "美的", "格力", "海尔", "耐克", "阿迪达斯", "安踏", "李宁",
            "雅诗兰黛", "兰蔻", "SK-II", "罗技", "雷蛇", "JBL", "BOSE",
        ]
        for brand in known_brands:
            if brand.lower() in title.lower():
                return brand
        return ""

    # ------------------------------------------------------------------
    # Crawl orchestration
    # ------------------------------------------------------------------

    async def crawl(self) -> List[Dict[str, Any]]:
        """
        Full crawl pipeline for JD.com.

        For each category:
          1. Open search result pages until quota or pages exhausted.
          2. For every product link found, visit the detail page.
          3. Extract and store structured data.

        Returns the aggregated product list.
        """
        logger.info(
            "Starting JD crawl: %d categories, max %d per category",
            len(self.categories),
            self.max_per_category,
        )

        try:
            await self._init_browser()
            page = await self._new_page()

            for cat_idx, category in enumerate(self.categories, 1):
                logger.info(
                    "[%d/%d] Crawling category: %s",
                    cat_idx,
                    len(self.categories),
                    category,
                )
                collected = 0

                for page_num in range(1, _MAX_PAGES_PER_CATEGORY + 1):
                    if collected >= self.max_per_category:
                        break

                    url = _SEARCH_URL.format(
                        keyword=quote(category),
                        page=page_num * 2 - 1,  # JD uses odd page numbers
                    )
                    ok = await self._fetch_page(page, url)
                    if not ok:
                        logger.warning(
                            "Failed to load search page %d for '%s'", page_num, category
                        )
                        continue

                    links = await self._extract_product_links(page)
                    if not links:
                        logger.info("No more links for '%s' on page %d", category, page_num)
                        break

                    for link in links:
                        if collected >= self.max_per_category:
                            break

                        ok = await self._fetch_page(page, link)
                        if not ok:
                            continue

                        product = await self._extract_product_detail(page, category)
                        if product:
                            self._append_product(product)
                            collected += 1
                            logger.debug(
                                "  [%d/%d] %s",
                                collected,
                                self.max_per_category,
                                product.get("title", "")[:50],
                            )

                logger.info(
                    "Category '%s' done — collected %d products", category, collected
                )

            self._save_products()

        finally:
            await self._close_browser()

        logger.info("JD crawl finished. Stats: %s", json.dumps(self.stats))
        return self._products
