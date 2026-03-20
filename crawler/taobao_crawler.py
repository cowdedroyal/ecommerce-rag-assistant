"""
Taobao / 1688 product crawler.

Primary target: 1688.com (Alibaba's domestic B2B marketplace) because it has
substantially lighter anti-bot measures than taobao.com while still offering
rich Chinese product data suitable for RAG training.

Fallback: if a ``use_taobao=True`` flag is passed the crawler will attempt
taobao.com search pages instead — but success is not guaranteed due to
Taobao's aggressive bot detection (slider captcha, login walls, etc.).

Output: data/crawled/taobao_products.json
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

# 1688 search URL template
_1688_SEARCH_URL = (
    "https://s.1688.com/selloffer/offer_search.htm"
    "?keywords={keyword}&beginPage={page}"
)

# Taobao search URL template (fallback)
_TAOBAO_SEARCH_URL = (
    "https://s.taobao.com/search?q={keyword}&s={offset}"
)

_MAX_PAGES_PER_CATEGORY = 4


class TaobaoCrawler(BaseCrawler):
    """
    Concrete crawler for 1688.com (default) or Taobao.

    Design mirrors JDCrawler: search → list → detail.
    All extracted records carry ``source="1688"`` or ``source="taobao"``.
    """

    def __init__(
        self,
        output_path: Optional[Path] = None,
        categories: Optional[List[str]] = None,
        max_per_category: int = 100,
        use_taobao: bool = False,
        **kwargs,
    ):
        output_path = output_path or Path("data/crawled/taobao_products.json")
        super().__init__(
            output_path=output_path,
            categories=categories or [],
            max_per_category=max_per_category,
            **kwargs,
        )
        self.use_taobao = use_taobao
        self.source = "taobao" if use_taobao else "1688"

    # ------------------------------------------------------------------
    # URL builders
    # ------------------------------------------------------------------

    def _search_url(self, keyword: str, page_num: int) -> str:
        if self.use_taobao:
            offset = (page_num - 1) * 44  # Taobao paginates by offset
            return _TAOBAO_SEARCH_URL.format(
                keyword=quote(keyword), offset=offset
            )
        return _1688_SEARCH_URL.format(
            keyword=quote(keyword), page=page_num
        )

    # ------------------------------------------------------------------
    # Search page — 1688
    # ------------------------------------------------------------------

    async def _extract_links_1688(self, page: Page) -> List[str]:
        """Extract product detail links from a 1688 search result page."""
        await self._scroll_page(page, scrolls=5)
        links: List[str] = []

        try:
            # 1688 renders offer cards in several possible container layouts
            selectors = [
                ".sm-offer-item a.sm-offer-link",
                ".offer-list-row a[href*='detail']",
                "a[href*='offer'][href*='1688.com']",
                ".img-container a",
            ]
            seen = set()
            for sel in selectors:
                anchors = await page.query_selector_all(sel)
                for anchor in anchors:
                    href = await anchor.get_attribute("href")
                    if not href:
                        continue
                    url = href if href.startswith("http") else f"https:{href}"
                    # Filter to actual detail pages
                    if "detail" in url or "offer" in url:
                        if url not in seen:
                            seen.add(url)
                            links.append(url)
        except Exception as exc:
            logger.warning("1688 link extraction error: %s", exc)

        logger.debug("1688 search page yielded %d links", len(links))
        return links

    # ------------------------------------------------------------------
    # Search page — Taobao
    # ------------------------------------------------------------------

    async def _extract_links_taobao(self, page: Page) -> List[str]:
        """Extract product links from a Taobao search result page."""
        await self._scroll_page(page, scrolls=5)
        links: List[str] = []

        try:
            selectors = [
                ".items .item a.J_ClickStat",
                ".m-itemlist .items .item a",
                "a[href*='item.taobao.com']",
                "a[href*='detail.tmall.com']",
            ]
            seen = set()
            for sel in selectors:
                anchors = await page.query_selector_all(sel)
                for anchor in anchors:
                    href = await anchor.get_attribute("href")
                    if not href:
                        continue
                    url = href if href.startswith("http") else f"https:{href}"
                    if url not in seen:
                        seen.add(url)
                        links.append(url)
        except Exception as exc:
            logger.warning("Taobao link extraction error: %s", exc)

        logger.debug("Taobao search page yielded %d links", len(links))
        return links

    # ------------------------------------------------------------------
    # Detail page — 1688
    # ------------------------------------------------------------------

    async def _extract_detail_1688(
        self, page: Page, category: str
    ) -> Optional[Dict[str, Any]]:
        """Parse a 1688 offer detail page into a structured dict."""
        product: Dict[str, Any] = {"source": self.source, "category": category}

        try:
            # product_id from URL
            url = page.url
            match = re.search(r"offer/(\d+)", url) or re.search(r"(\d+)\.html", url)
            product["product_id"] = match.group(1) if match else url

            # title
            title_el = (
                await page.query_selector(".title-text")
                or await page.query_selector("h1.d-title")
                or await page.query_selector(".mod-detail-title")
            )
            product["title"] = (
                (await title_el.inner_text()).strip() if title_el else ""
            )

            # brand
            product["brand"] = await self._extract_brand_1688(page, product.get("title", ""))

            # price
            product["price"] = await self._extract_price_1688(page)

            # rating & review_count (1688 often lacks ratings; use trade count)
            product["rating"] = None
            product["review_count"] = await self._extract_trade_count(page)

            # description
            product["description"] = await self._extract_description_1688(page)

            # specs
            product["specs"] = await self._extract_specs_1688(page)

            # features
            product["features"] = await self._extract_features_1688(page)

        except Exception as exc:
            logger.error("1688 detail extraction error: %s", exc)
            return None

        if not product.get("title"):
            return None
        return product

    # ------------------------------------------------------------------
    # Detail page — Taobao
    # ------------------------------------------------------------------

    async def _extract_detail_taobao(
        self, page: Page, category: str
    ) -> Optional[Dict[str, Any]]:
        """Parse a Taobao / Tmall item detail page."""
        product: Dict[str, Any] = {"source": self.source, "category": category}

        try:
            url = page.url
            match = re.search(r"id=(\d+)", url) or re.search(r"/(\d+)\.html", url)
            product["product_id"] = match.group(1) if match else url

            # title
            title_el = (
                await page.query_selector("h3[data-title]")
                or await page.query_selector(".tb-main-title")
                or await page.query_selector(".ItemHeader--mainTitle")
            )
            product["title"] = (
                (await title_el.inner_text()).strip() if title_el else ""
            )

            # brand
            product["brand"] = await self._extract_brand_taobao(
                page, product.get("title", "")
            )

            # price
            product["price"] = await self._extract_price_taobao(page)

            # rating
            rating_el = await page.query_selector(".tb-rating-count, .Rating--count")
            if rating_el:
                text = await rating_el.inner_text()
                digits = re.sub(r"[^\d.]", "", text)
                try:
                    product["rating"] = min(float(digits), 5.0)
                except ValueError:
                    product["rating"] = None
            else:
                product["rating"] = None

            # review_count
            rc_el = await page.query_selector(
                ".tb-rate-count, .Rating--rateCount"
            )
            if rc_el:
                text = await rc_el.inner_text()
                digits = re.sub(r"[^\d]", "", text.replace("万", "0000"))
                product["review_count"] = int(digits) if digits else None
            else:
                product["review_count"] = None

            # description
            desc_el = await page.query_selector(
                ".tb-subtitle, .ItemHeader--subTitle"
            )
            product["description"] = (
                (await desc_el.inner_text()).strip() if desc_el else ""
            )

            # specs
            product["specs"] = await self._extract_specs_taobao(page)

            # features
            product["features"] = await self._extract_features_taobao(page)

        except Exception as exc:
            logger.error("Taobao detail extraction error: %s", exc)
            return None

        if not product.get("title"):
            return None
        return product

    # ------------------------------------------------------------------
    # 1688 field helpers
    # ------------------------------------------------------------------

    async def _extract_price_1688(self, page: Page) -> Optional[float]:
        selectors = [
            ".price-text",
            ".d-price .value",
            ".mod-detail-price .value",
        ]
        for sel in selectors:
            el = await page.query_selector(sel)
            if el:
                text = await el.inner_text()
                cleaned = re.sub(r"[^\d.]", "", text.split("-")[0])
                try:
                    return float(cleaned)
                except ValueError:
                    continue
        return None

    async def _extract_brand_1688(self, page: Page, title: str) -> str:
        try:
            rows = await page.query_selector_all(
                ".obj-content .obj-sku tr, .detail-attr li"
            )
            for row in rows:
                text = (await row.inner_text()).strip()
                if "品牌" in text and ("：" in text or ":" in text):
                    sep = "：" if "：" in text else ":"
                    return text.split(sep, 1)[1].strip()
        except Exception:
            pass
        return self._infer_brand(title)

    async def _extract_trade_count(self, page: Page) -> Optional[int]:
        selectors = [".trade-count", ".d-trade-count", ".sale-count"]
        for sel in selectors:
            el = await page.query_selector(sel)
            if el:
                text = await el.inner_text()
                digits = re.sub(r"[^\d]", "", text.replace("万", "0000"))
                if digits:
                    return int(digits)
        return None

    async def _extract_description_1688(self, page: Page) -> str:
        selectors = [
            ".mod-detail-desc",
            ".offer-attr-list",
            ".detail-desc",
        ]
        for sel in selectors:
            el = await page.query_selector(sel)
            if el:
                text = (await el.inner_text()).strip()
                if text:
                    return text[:500]
        return ""

    async def _extract_specs_1688(self, page: Page) -> Dict[str, str]:
        specs: Dict[str, str] = {}
        try:
            items = await page.query_selector_all(
                ".obj-content .obj-sku tr, .detail-attr-item"
            )
            for item in items:
                text = (await item.inner_text()).strip()
                for sep in ["：", ":"]:
                    if sep in text:
                        k, v = text.split(sep, 1)
                        specs[k.strip()] = v.strip()
                        break
        except Exception as exc:
            logger.debug("1688 spec extraction issue: %s", exc)
        return specs

    async def _extract_features_1688(self, page: Page) -> List[str]:
        features: List[str] = []
        try:
            tags = await page.query_selector_all(
                ".detail-feature li, .offer-attr-item"
            )
            for tag in tags:
                text = (await tag.inner_text()).strip()
                if text and len(text) > 1:
                    features.append(text)
        except Exception:
            pass
        return features[:20]

    # ------------------------------------------------------------------
    # Taobao field helpers
    # ------------------------------------------------------------------

    async def _extract_price_taobao(self, page: Page) -> Optional[float]:
        selectors = [
            ".tb-rmb-num",
            ".Price--priceText",
            "span.price",
        ]
        for sel in selectors:
            el = await page.query_selector(sel)
            if el:
                text = await el.inner_text()
                cleaned = re.sub(r"[^\d.]", "", text.split("-")[0])
                try:
                    return float(cleaned)
                except ValueError:
                    continue
        return None

    async def _extract_brand_taobao(self, page: Page, title: str) -> str:
        try:
            attrs = await page.query_selector_all(".attributes-list li")
            for li in attrs:
                text = (await li.inner_text()).strip()
                if "品牌" in text:
                    for sep in ["：", ":"]:
                        if sep in text:
                            return text.split(sep, 1)[1].strip()
        except Exception:
            pass
        return self._infer_brand(title)

    async def _extract_specs_taobao(self, page: Page) -> Dict[str, str]:
        specs: Dict[str, str] = {}
        try:
            items = await page.query_selector_all(
                ".attributes-list li, .Attrs--attr"
            )
            for item in items:
                text = (await item.inner_text()).strip()
                for sep in ["：", ":"]:
                    if sep in text:
                        k, v = text.split(sep, 1)
                        specs[k.strip()] = v.strip()
                        break
        except Exception as exc:
            logger.debug("Taobao spec extraction: %s", exc)
        return specs

    async def _extract_features_taobao(self, page: Page) -> List[str]:
        features: List[str] = []
        try:
            tags = await page.query_selector_all(
                ".tb-property li, .ItemHeader--tag"
            )
            for tag in tags:
                text = (await tag.inner_text()).strip()
                if text and len(text) > 1:
                    features.append(text)
        except Exception:
            pass
        return features[:20]

    # ------------------------------------------------------------------
    # Shared utility
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_brand(title: str) -> str:
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

    async def _extract_links(self, page: Page) -> List[str]:
        """Dispatch to the correct search-page parser."""
        if self.use_taobao:
            return await self._extract_links_taobao(page)
        return await self._extract_links_1688(page)

    async def _extract_detail(
        self, page: Page, category: str
    ) -> Optional[Dict[str, Any]]:
        """Dispatch to the correct detail-page parser."""
        if self.use_taobao:
            return await self._extract_detail_taobao(page, category)
        return await self._extract_detail_1688(page, category)

    async def crawl(self) -> List[Dict[str, Any]]:
        """
        Full crawl pipeline for 1688 / Taobao.

        Follows the same search → list → detail pattern as JDCrawler.
        """
        target = "Taobao" if self.use_taobao else "1688"
        logger.info(
            "Starting %s crawl: %d categories, max %d per category",
            target,
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

                    url = self._search_url(category, page_num)
                    ok = await self._fetch_page(page, url)
                    if not ok:
                        logger.warning(
                            "Failed to load search page %d for '%s'",
                            page_num,
                            category,
                        )
                        continue

                    links = await self._extract_links(page)
                    if not links:
                        logger.info(
                            "No more links for '%s' on page %d",
                            category,
                            page_num,
                        )
                        break

                    for link in links:
                        if collected >= self.max_per_category:
                            break

                        ok = await self._fetch_page(page, link)
                        if not ok:
                            continue

                        product = await self._extract_detail(page, category)
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
                    "Category '%s' done — collected %d products",
                    category,
                    collected,
                )

            self._save_products()

        finally:
            await self._close_browser()

        logger.info("%s crawl finished. Stats: %s", target, json.dumps(self.stats))
        return self._products
