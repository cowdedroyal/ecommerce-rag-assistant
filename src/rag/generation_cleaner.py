import re


STRUCTURED_END_MARKERS = (
    "以上为本次推荐结果。",
    "以上为本轮推荐结果。",
    "以上为本次对比结论。",
    "以上为本轮对比结论。",
    "以上为本次结论。",
    "以上为本轮结论。",
    "以上为本次回答。",
    "以上为本轮回答。",
)

HARD_STOP_MARKERS = ("rumpe", "spep", "+lsi", "𬭩user", "𬭩assistant", "RGBO")
INLINE_NOISE_TOKENS = ("ropic", "玭", "🧐", "🖇", "垞", "�")
LEAKED_ROLE_PATTERN = re.compile(r"(?:^|\n)(?:[^\n]{0,3})?(?:user|assistant)\n", re.IGNORECASE)
LIST_ITEM_PATTERN = re.compile(r"^\s*(?:先看\s*)?\[(\d+)\]\s*(.+)$")
PRICE_PATTERN = re.compile(r"价格\s*[:：]?\s*(?:¥|￥)?\s*(\d+(?:\.\d+)?)\s*元?")
RECOMMENDATION_LINE_PATTERN = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
EMPATHY_PATTERN = re.compile(r"(抱歉|理解|麻烦|先别着急|我先帮您)")
ITEM_TITLE_PATTERN = re.compile(r"^\s*\d+\.\s+\[(\d+)\]\s+([^，。\n]+)")
TRUNCATED_RECOMMENDATION_PATTERN = re.compile(r"(推荐\s+\[(\d+)\]\s+)([^。\n]+)$")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")


def sanitize_generated_text(text: str) -> str:
    cleaned = str(text).replace("\r\n", "\n").strip()
    cleaned = _trim_at_explicit_end_markers(cleaned)
    cleaned = _trim_at_hard_stop_markers(cleaned)
    cleaned = _trim_leaked_roles(cleaned)
    cleaned = _strip_inline_noise(cleaned)
    cleaned = _strip_markdown_links(cleaned)
    cleaned = _normalize_recommendation_prices(cleaned)
    cleaned = _normalize_numbered_recommendation_lines(cleaned)
    cleaned = _dedupe_consecutive_lines(cleaned)
    cleaned = _dedupe_repeated_sentences(cleaned)
    cleaned = _ensure_service_intro_for_recommendations(cleaned)
    cleaned = _expand_truncated_recommendation_reference(cleaned)
    cleaned = _finish_sentence(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _trim_at_explicit_end_markers(text: str) -> str:
    positions = []
    for marker in STRUCTURED_END_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            positions.append((idx, marker))
    if not positions:
        return text
    idx, marker = min(positions, key=lambda item: item[0])
    return text[: idx + len(marker)].rstrip()


def _trim_at_hard_stop_markers(text: str) -> str:
    cleaned = text
    for marker in HARD_STOP_MARKERS:
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx].rstrip()
    return cleaned


def _trim_leaked_roles(text: str) -> str:
    leaked_role_match = LEAKED_ROLE_PATTERN.search(text)
    if leaked_role_match:
        return text[: leaked_role_match.start()].rstrip()
    return text


def _strip_inline_noise(text: str) -> str:
    cleaned = text
    for token in INLINE_NOISE_TOKENS:
        cleaned = cleaned.replace(token, "")

    lines = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        signal = re.sub(r"[\W_]+", "", line, flags=re.UNICODE)
        if not signal:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _strip_markdown_links(text: str) -> str:
    return MARKDOWN_LINK_PATTERN.sub(lambda match: match.group(1), text)


def _normalize_recommendation_prices(text: str) -> str:
    return PRICE_PATTERN.sub(lambda match: f"价格：¥{match.group(1)}", text)


def _normalize_numbered_recommendation_lines(text: str) -> str:
    normalized_lines = []
    for line in text.splitlines():
        match = LIST_ITEM_PATTERN.match(line)
        if match:
            idx = match.group(1)
            content = match.group(2).strip()
            normalized_lines.append(f"{idx}. [{idx}] {content}")
            continue
        normalized_lines.append(line.rstrip())
    return "\n".join(normalized_lines).strip()


def _dedupe_consecutive_lines(text: str) -> str:
    deduped_lines = []
    previous = None
    for line in text.splitlines():
        current = line.strip()
        if not current:
            if deduped_lines and deduped_lines[-1] != "":
                deduped_lines.append("")
            continue
        if current == previous:
            continue
        deduped_lines.append(current)
        previous = current
    return "\n".join(deduped_lines).strip()


def _ensure_service_intro_for_recommendations(text: str) -> str:
    if EMPATHY_PATTERN.search(text):
        return text
    item_count = len(RECOMMENDATION_LINE_PATTERN.findall(text))
    if item_count < 2:
        return text
    if "[1]" not in text and "[2]" not in text:
        return text
    return f"我先帮您按当前需求筛 {item_count} 款：\n\n{text}".strip()


def _dedupe_repeated_sentences(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        parts = [part for part in re.findall(r"[^。！？!?]+[。！？!?]?", stripped) if part.strip()]
        if len(parts) <= 1:
            cleaned_lines.append(stripped)
            continue

        deduped_parts = []
        previous = ""
        for part in parts:
            normalized = re.sub(r"\s+", "", part).rstrip("。！？!?")
            if not normalized:
                continue
            if normalized == previous:
                continue
            if len(normalized) <= 6 and previous.startswith(normalized):
                continue
            deduped_parts.append(part.strip())
            previous = normalized
        cleaned_lines.append("".join(deduped_parts).strip())
    return "\n".join(cleaned_lines).strip()


def _expand_truncated_recommendation_reference(text: str) -> str:
    item_titles = {}
    for line in text.splitlines():
        match = ITEM_TITLE_PATTERN.match(line)
        if match:
            item_titles[match.group(1)] = match.group(2).strip()

    tail_match = TRUNCATED_RECOMMENDATION_PATTERN.search(text)
    if not tail_match:
        return text

    item_idx = tail_match.group(2)
    partial_title = tail_match.group(3).strip()
    full_title = item_titles.get(item_idx, "")
    if full_title and len(partial_title) < len(full_title) and full_title.startswith(partial_title):
        return f"{text[:tail_match.start(3)]}{full_title}"
    return text


def _finish_sentence(text: str) -> str:
    if not text:
        return text
    if text[-1] in "。！？!?)）】]}」』":
        return text
    return f"{text}。"
