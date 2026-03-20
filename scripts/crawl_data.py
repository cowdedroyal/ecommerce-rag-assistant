#!/usr/bin/env python3
"""
CLI entry-point for the e-commerce product crawl pipeline.

Usage examples::

    # Crawl all sources with default categories
    python scripts/crawl_data.py --source all

    # Crawl JD only, specific categories, 50 items each
    python scripts/crawl_data.py --source jd --categories 手机 耳机 --max-per-category 50

    # Crawl Taobao (1688) and clean
    python scripts/crawl_data.py --source taobao

The script will:
  1. Launch the selected crawler(s).
  2. Persist raw JSON to ``data/crawled/``.
  3. Run the data cleaner to produce ``data/processed/products_zh.csv``.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# Ensure project root is on sys.path so ``crawler`` package is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from crawler.jd_crawler import JDCrawler
from crawler.taobao_crawler import TaobaoCrawler
from crawler.data_cleaner import DataCleaner
from src.config import Settings

# ---------------------------------------------------------------------------
# Logging setup (Rich-powered)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("crawl_data")

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_categories(categories: tuple, settings: Settings) -> List[str]:
    """Return explicit categories or fall back to config defaults."""
    if categories:
        return list(categories)
    return list(settings.CRAWL_CATEGORIES)


def _print_summary_table(source: str, stats: dict, elapsed: float) -> None:
    """Render a Rich table summarising crawl results."""
    table = Table(title=f"Crawl Summary — {source}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Total requests", str(stats.get("total_requests", 0)))
    table.add_row("Successful", str(stats.get("successful", 0)))
    table.add_row("Failed", str(stats.get("failed", 0)))
    table.add_row("Retried", str(stats.get("retried", 0)))
    table.add_row(
        "Success rate",
        f"{stats.get('success_rate', 0) * 100:.1f}%",
    )
    table.add_row("Elapsed", f"{elapsed:.1f} s")

    console.print(table)


# ---------------------------------------------------------------------------
# Async crawl runners
# ---------------------------------------------------------------------------

async def _run_jd(
    categories: List[str],
    max_per_category: int,
    settings: Settings,
) -> dict:
    """Launch the JD crawler and return stats."""
    crawler = JDCrawler(
        output_path=settings.CRAWLED_DATA_DIR / "jd_products.json",
        categories=categories,
        max_per_category=max_per_category,
        delay_min=settings.CRAWL_DELAY_MIN,
        delay_max=settings.CRAWL_DELAY_MAX,
        max_retries=settings.CRAWL_MAX_RETRIES,
    )
    await crawler.crawl()
    return crawler.stats


async def _run_taobao(
    categories: List[str],
    max_per_category: int,
    settings: Settings,
) -> dict:
    """Launch the Taobao/1688 crawler and return stats."""
    crawler = TaobaoCrawler(
        output_path=settings.CRAWLED_DATA_DIR / "taobao_products.json",
        categories=categories,
        max_per_category=max_per_category,
        delay_min=settings.CRAWL_DELAY_MIN,
        delay_max=settings.CRAWL_DELAY_MAX,
        max_retries=settings.CRAWL_MAX_RETRIES,
    )
    await crawler.crawl()
    return crawler.stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--source",
    type=click.Choice(["jd", "taobao", "all"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Which data source(s) to crawl.",
)
@click.option(
    "--categories",
    multiple=True,
    help="Category keywords to crawl (repeatable). Defaults to config list.",
)
@click.option(
    "--max-per-category",
    type=int,
    default=100,
    show_default=True,
    help="Maximum number of products to collect per category.",
)
@click.option(
    "--no-clean",
    is_flag=True,
    default=False,
    help="Skip the data cleaning step after crawling.",
)
def main(
    source: str,
    categories: tuple,
    max_per_category: int,
    no_clean: bool,
) -> None:
    """Crawl e-commerce product data and produce a cleaned CSV."""

    settings = Settings()
    cats = _resolve_categories(categories, settings)

    console.print(
        Panel.fit(
            f"[bold]E-commerce Product Crawler[/bold]\n"
            f"Source: [cyan]{source}[/cyan]  |  "
            f"Categories: [cyan]{len(cats)}[/cyan]  |  "
            f"Max/category: [cyan]{max_per_category}[/cyan]",
            border_style="blue",
        )
    )
    console.print(f"Categories: {', '.join(cats)}\n")

    # ------------------------------------------------------------------
    # Phase 1: Crawl
    # ------------------------------------------------------------------
    t_start = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        if source in ("jd", "all"):
            task = progress.add_task("[cyan]Crawling JD.com ...", total=None)
            jd_stats = asyncio.run(_run_jd(cats, max_per_category, settings))
            progress.update(task, completed=True, description="[green]JD.com done")
            _print_summary_table("JD", jd_stats, jd_stats.get("elapsed_seconds", 0))

        if source in ("taobao", "all"):
            task = progress.add_task("[cyan]Crawling 1688/Taobao ...", total=None)
            tb_stats = asyncio.run(_run_taobao(cats, max_per_category, settings))
            progress.update(task, completed=True, description="[green]1688/Taobao done")
            _print_summary_table("1688/Taobao", tb_stats, tb_stats.get("elapsed_seconds", 0))

    t_crawl = time.time() - t_start
    console.print(f"\nCrawl phase finished in [bold]{t_crawl:.1f}[/bold] seconds.\n")

    # ------------------------------------------------------------------
    # Phase 2: Clean
    # ------------------------------------------------------------------
    if no_clean:
        console.print("[yellow]Skipping data cleaning (--no-clean).[/yellow]")
        return

    console.print("[cyan]Running data cleaner ...[/cyan]")

    cleaner = DataCleaner(
        input_dir=settings.CRAWLED_DATA_DIR,
        output_path=settings.PROCESSED_DATA_DIR / "products_zh.csv",
    )
    df = cleaner.run()

    console.print(
        Panel.fit(
            f"[bold green]Cleaning complete[/bold green]\n"
            f"Products: [cyan]{len(df)}[/cyan]\n"
            f"Output:   [cyan]{cleaner.output_path}[/cyan]",
            border_style="green",
        )
    )

    total_elapsed = time.time() - t_start
    console.print(f"\nTotal pipeline time: [bold]{total_elapsed:.1f}[/bold] seconds.")


if __name__ == "__main__":
    main()
