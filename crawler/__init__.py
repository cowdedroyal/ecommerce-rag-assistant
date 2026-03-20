"""
Crawler package for e-commerce product data collection.

Supports JD (jd.com) and Taobao/1688 as data sources.
"""

from crawler.base import BaseCrawler
from crawler.jd_crawler import JDCrawler
from crawler.taobao_crawler import TaobaoCrawler
from crawler.data_cleaner import DataCleaner

__all__ = ["BaseCrawler", "JDCrawler", "TaobaoCrawler", "DataCleaner"]
