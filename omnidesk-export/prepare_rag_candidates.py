#!/usr/bin/env python3
"""Create a safer intermediate dataset for reviewing OmniDesk conversations.

This script does not modify the original dataset. It excludes internal notes and
unknown roles, removes attachment metadata, applies conservative regex-based
redaction, and writes aggregate statistics only (never matched values).

IMPORTANT: This is a preprocessing aid, not a guarantee of anonymization and
not a production-ready knowledge base. Review the output before indexing it.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("output/dataset.jsonl")
DEFAULT_OUTPUT = Path("output/rag_candidates.jsonl")
DEFAULT_REPORT = Path("output/rag_prepare_report.json")

# These patterns are intentionally conservative. They reduce common risks but
# cannot detect every personal detail (for example, names or postal addresses).
PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "url": re.compile(r"\bhttps?://[^\s<>\"']+", re.IGNORECASE),
    "long_number": re.compile(r"(?<!\d)\d{8,}(?!\d)"),
    "secret_assignment": re.compile(
        r"\b(password|passwd|passcode|token|api[ _-]?key|secret|парол[ьяе]?|токен)\b"
        r"\s*(?:is\s+|есть\s+|[:=]\s*)[^\s,;]+",
        re.IGNORECASE,
    ),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
    "private_key_block": re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.IGNORECASE | re.DOTALL,
    ),
}

PHONE_CANDIDATE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")


class VisibleTextParser(HTMLParser):
    """Extract basic visible text from HTML without external dependencies."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        cleaned = data.strip()
        if cleaned:
            self.parts.append(cleaned)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"br", "p", "div", "li", "tr", "hr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"p", "div", "li", "tr"}:
            self.parts.append("\n")


def html_to_text(value: str) -> str:
    parser = VisibleTextParser()
    try:
        parser.feed(value)
        parser.close()
        result = " ".join(parser.parts)
    except Exception:
        result = re.sub(r"<[^>]*>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(result)).strip()


def is_truthy(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def mask_phone(match: re.Match[str]) -> str:
    value = match.group(0)
    digits = re.sub(r"\D", "", value)
    # Avoid masking short numbers and dates; mask phone-like lengths only.
    if 10 <= len(digits) <= 15:
        return "[PHONE]"
    return value


def redact_text(value: Any, counts: Counter[str]) -> str:
    if not isinstance(value, str):
        return ""
    text = value

    # Redact URLs first: query parameters can contain words such as token=...,
    # and those placeholders should not be mistaken for trailing URL punctuation.
    url_pattern = PATTERNS["url"]
    url_matches = list(url_pattern.finditer(text))
    if url_matches:
        counts["url"] += len(url_matches)

        def replace_url(match: re.Match[str]) -> str:
            found = match.group(0)
            suffix = ""
            while found and found[-1] in ".,;:!?)]}":
                suffix = found[-1] + suffix
                found = found[:-1]
            return "[URL]" + suffix

        text = url_pattern.sub(replace_url, text)

    # Redact credentials and key material before the remaining common patterns.
    for name in ("private_key_block", "bearer_token", "secret_assignment"):
        pattern = PATTERNS[name]
        matches = list(pattern.finditer(text))
        if matches:
            counts[name] += len(matches)
            if name == "secret_assignment":
                text = pattern.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
            elif name == "bearer_token":
                text = pattern.sub("Bearer [REDACTED]", text)
            else:
                text = pattern.sub("[PRIVATE KEY REDACTED]", text)

    # Mask emails, IPs and long numbers after URLs and secrets have been removed.
    for name in ("email", "ip_address", "long_number"):
        pattern = PATTERNS[name]
        matches = list(pattern.finditer(text))
        if matches:
            counts[name] += len(matches)
            replacement = {
                "email": "[EMAIL]",
                "ip_address": "[IP_ADDRESS]",
                "long_number": "[LONG_NUMBER]",
                "url": "[URL]",
            }[name]
            text = pattern.sub(replacement, text)

    phone_matches = list(PHONE_CANDIDATE.finditer(text))
    actual_phone_matches = 0
    for match in reversed(phone_matches):
        replacement = mask_phone(match)
        if replacement == "[PHONE]":
            actual_phone_matches += 1
            text = text[:match.start()] + replacement + text[match.end():]
    counts["phone"] += actual_phone_matches

    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def get_message_text(message: dict[str, Any], counts: Counter[str]) -> str:
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        raw = content
    else:
        content_html = message.get("content_html")
        raw = html_to_text(content_html) if isinstance(content_html, str) else ""
    return redact_text(raw, counts)


def safe_label_values(value: Any, counts: Counter[str]) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = [item for item in value if isinstance(item, str)]
    else:
        values = []
    return [cleaned for item in values if (cleaned := redact_text(item, counts))]


def process_record(record: dict[str, Any], stats: Counter[str], redactions: Counter[str]) -> dict[str, Any] | None:
    case_id = record.get("case_id")
    if case_id is None:
        stats["records_missing_case_id"] += 1
        return None

    subject = redact_text(record.get("subject") or "", redactions)
    category_value = record.get("category")
    category = redact_text(category_value, redactions) if isinstance(category_value, str) else ""
    labels = safe_label_values(record.get("labels", []), redactions)

    source_messages = record.get("messages")
    if not isinstance(source_messages, list):
        source_messages = []

    candidate_messages: list[dict[str, str]] = []

    for message in source_messages:
        if not isinstance(message, dict):
            stats["invalid_message_objects"] += 1
            continue

        role = message.get("role")
        role = role.strip().lower() if isinstance(role, str) else "unknown"
        stats[f"source_role_{role or 'unknown'}"] += 1

        if role == "internal_note" or is_truthy(message.get("note")):
            stats["internal_notes_excluded"] += 1
            continue

        if role not in {"customer", "support"}:
            stats["unknown_or_other_roles_excluded"] += 1
            continue

        text = get_message_text(message, redactions)
        if not text:
            stats["messages_without_text_excluded"] += 1
            if message.get("attachments"):
                stats["messages_with_attachments_but_no_text_excluded"] += 1
            continue

        candidate_messages.append({"role": role, "text": text})
        stats[f"included_role_{role}"] += 1

        attachments = message.get("attachments")
        if isinstance(attachments, list):
            stats["attachment_items_omitted"] += len(attachments)
        elif isinstance(attachments, dict) and attachments:
            stats["attachment_items_omitted"] += 1

    if not candidate_messages:
        stats["cases_without_eligible_messages_excluded"] += 1
        return None

    stats["cases_included"] += 1
    stats["included_messages"] += len(candidate_messages)

    # case_id is kept only for local traceability during review. Do not expose it
    # to the chatbot or carry it into a public/user-facing retrieval result.
    return {
        "source_case_id": str(case_id),
        "subject": subject,
        "category": category,
        "labels": labels,
        "created_at": record.get("created_at"),
        "messages": candidate_messages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a PII-redacted intermediate dataset for RAG review."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the candidate output and report, but never the input.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Не найден исходный файл: {args.input}")
        return 1

    try:
        if args.input.resolve() == args.output.resolve():
            print("Ошибка: входной и выходной файлы должны быть разными.")
            return 1
    except OSError:
        pass

    if not args.overwrite and (args.output.exists() or args.report.exists()):
        print(
            "Выходной файл или отчёт уже существует. Ничего не изменено. "
            "Укажи другие пути или добавь --overwrite, если хочешь заменить именно их."
        )
        return 1

    stats: Counter[str] = Counter()
    redactions: Counter[str] = Counter()
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    malformed_lines = 0
    valid_records = 0
    nonempty_lines = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file first, so a failed run won't leave a half-written output.
    temp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        with args.input.open("r", encoding="utf-8") as source, temp_output.open(
            "w", encoding="utf-8"
        ) as output:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                nonempty_lines += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue
                if not isinstance(record, dict):
                    malformed_lines += 1
                    continue

                valid_records += 1
                case_id = record.get("case_id")
                if case_id is not None:
                    case_key = str(case_id)
                    if case_key in seen_ids:
                        duplicate_ids.add(case_key)
                        stats["duplicate_records_skipped"] += 1
                        continue
                    seen_ids.add(case_key)

                candidate = process_record(record, stats, redactions)
                if candidate is not None:
                    output.write(json.dumps(candidate, ensure_ascii=False) + "\n")

        temp_output.replace(args.output)
    except Exception:
        if temp_output.exists():
            temp_output.unlink()
        raise

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(args.input),
        "candidate_file": str(args.output),
        "source_nonempty_lines": nonempty_lines,
        "valid_source_records": valid_records,
        "malformed_source_lines": malformed_lines,
        "unique_source_case_ids": len(seen_ids),
        "duplicate_case_ids": len(duplicate_ids),
        "duplicate_records_skipped": stats["duplicate_records_skipped"],
        "cases_included": stats["cases_included"],
        "cases_without_eligible_messages_excluded": stats["cases_without_eligible_messages_excluded"],
        "source_role_counts": {
            key.removeprefix("source_role_"): value
            for key, value in sorted(stats.items())
            if key.startswith("source_role_")
        },
        "included_role_counts": {
            key.removeprefix("included_role_"): value
            for key, value in sorted(stats.items())
            if key.startswith("included_role_")
        },
        "internal_notes_excluded": stats["internal_notes_excluded"],
        "unknown_or_other_roles_excluded": stats["unknown_or_other_roles_excluded"],
        "messages_without_text_excluded": stats["messages_without_text_excluded"],
        "messages_with_attachments_but_no_text_excluded": stats["messages_with_attachments_but_no_text_excluded"],
        "attachment_items_omitted": stats["attachment_items_omitted"],
        "included_messages": stats["included_messages"],
        "regex_redaction_counts": dict(sorted(redactions.items())),
        "note": (
            "This is an intermediate review dataset, not a production RAG index. "
            "Regex redaction is approximate and does not guarantee anonymization. "
            "Internal notes, unknown-role messages, and attachment metadata are excluded. "
            "Review the output before indexing it; names, addresses, contextual identifiers, "
            "and sensitive facts may remain. source_case_id is for local traceability only."
        ),
    }

    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nКандидаты сохранены: {args.output}")
    print(f"Сводный отчёт сохранён: {args.report}")
    print("Исходный файл не изменялся. Не индексируй этот файл без дополнительной проверки.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
