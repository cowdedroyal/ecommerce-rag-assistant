#!/usr/bin/env python3
"""
Repair pairwise judge results when the raw rationale still contains JSON-ish fields.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark import Benchmark


def _parse_failure_tags(text: str) -> list[str]:
    match = re.search(r'"failure_tags"\s*:\s*\[(.*?)\]', text, re.S)
    if not match:
        return []
    return [tag.strip() for tag in re.findall(r'"([^"]+)"', match.group(1)) if tag.strip()]


def _repair_pairwise_entry(item: dict) -> dict:
    text = str(item.get("rationale", ""))
    winner_match = re.search(r'"winner"\s*:\s*"(a|b|tie)"', text)
    if not winner_match:
        return item

    winner = winner_match.group(1)
    confidence_match = re.search(r'"confidence"\s*:\s*"(low|medium|high)"', text)
    rationale_match = re.search(r'"rationale"\s*:\s*"([^"]+)"', text)

    item["winner"] = winner
    item["winner_stage"] = item["stage_a"] if winner == "a" else item["stage_b"] if winner == "b" else "tie"
    if confidence_match:
        item["confidence"] = confidence_match.group(1)
    if rationale_match:
        item["rationale"] = rationale_match.group(1)
    item["failure_tags"] = _parse_failure_tags(text)
    return item


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: repair_pairwise_report.py <src_json> <dst_prefix>")

    src = Path(sys.argv[1])
    dst_prefix = Path(sys.argv[2])
    dst_json = dst_prefix.with_suffix(".json")
    dst_md = dst_prefix.with_suffix(".md")

    data = json.loads(src.read_text(encoding="utf-8"))

    pairwise_summary = data.get("pairwise_judging", {})
    comparisons = [_repair_pairwise_entry(dict(item)) for item in pairwise_summary.get("comparisons", [])]
    pairwise_summary["comparisons"] = comparisons
    pairwise_summary["winner_count"] = dict(
        Counter(
            item["winner_stage"]
            for item in comparisons
            if item.get("winner_stage") and item.get("winner_stage") != "tie"
        )
    )

    for route in data.get("routing", []):
        pairwise = route.get("pairwise", {})
        for key, item in list(pairwise.items()):
            pairwise[key] = _repair_pairwise_entry(dict(item))

    bench = Benchmark.__new__(Benchmark)
    bench.cases = [SimpleNamespace(**case) for case in data.get("cases", [])]
    report = Benchmark._render_markdown_report(bench, data)

    dst_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    dst_md.write_text(report, encoding="utf-8")

    print(dst_json)
    print(dst_md)
    print(json.dumps(pairwise_summary.get("winner_count", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
