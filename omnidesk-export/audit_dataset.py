#!/usr/bin/env python3
"""Read-only audit for an OmniDesk JSONL export.

The script does not modify dataset.jsonl and never prints message text or
matched personal data. It writes aggregate results to output/audit_report.json.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATASET_PATH = Path("output/dataset.jsonl")
REPORT_PATH = Path("output/audit_report.json")

# These are heuristic indicators, not a complete PII detector.
PATTERNS = {
    "email_like_fields": re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
    ),
    "phone_like_fields": re.compile(
        r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)"
    ),
    "ip_address_like_fields": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
    "url_like_fields": re.compile(r"\bhttps?://[^\s<>\"']+", re.IGNORECASE),
    "secret_keyword_fields": re.compile(
        r"\b(password|passwd|token|api[ _-]?key|secret|bearer)\b"
        r"|парол[ьяею]|токен|секретн[а-я]*",
        re.IGNORECASE,
    ),
    "long_number_like_fields": re.compile(r"(?<!\d)\d{8,}(?!\d)"),
}


def truthy_note(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def message_texts(message: dict[str, Any]) -> list[str]:
    """Return text-bearing fields only; never include IDs or timestamps."""
    texts: list[str] = []
    for key in ("content", "content_html"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value)

    # Attachment metadata may contain a filename or URL; scan strings in it,
    # but do not scan arbitrary IDs and timestamps in the message object.
    attachments = message.get("attachments")

    def walk(value: Any) -> None:
        if isinstance(value, str):
            if value.strip():
                texts.append(value)
        elif isinstance(value, list):
            for part in value:
                walk(part)
        elif isinstance(value, dict):
            for key, part in value.items():
                if key.lower().endswith("_id") or key.lower() in {
                    "id", "created_at", "updated_at", "sent_at", "timestamp"
                }:
                    continue
                walk(part)

    if attachments is not None:
        walk(attachments)
    return texts


def main() -> int:
    if not DATASET_PATH.exists():
        print(f"Не найден файл {DATASET_PATH}. Запусти скрипт из корня проекта.")
        return 1

    total_lines = 0
    malformed_lines = 0
    valid_records = 0
    missing_case_id_records = 0
    cases_without_messages = 0
    cases_with_internal_notes = 0
    total_messages = 0
    messages_missing_content = 0
    messages_missing_created_at = 0
    messages_with_attachments = 0
    total_attachments = 0

    seen_case_ids: set[str] = set()
    duplicate_case_ids: set[str] = set()
    message_roles: Counter[str] = Counter()
    pattern_field_counts: Counter[str] = Counter()
    fields_checked = 0
    messages_with_suspected_patterns: set[tuple[str, str]] = set()

    with DATASET_PATH.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            total_lines += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue

            if not isinstance(record, dict):
                malformed_lines += 1
                continue

            valid_records += 1
            case_id_value = record.get("case_id")
            if case_id_value is None:
                missing_case_id_records += 1
                case_id = f"<missing-line-{line_number}>"
            else:
                case_id = str(case_id_value)
                if case_id in seen_case_ids:
                    duplicate_case_ids.add(case_id)
                seen_case_ids.add(case_id)

            subject = record.get("subject")
            if isinstance(subject, str) and subject.strip():
                fields_checked += 1
                for pattern_name, pattern in PATTERNS.items():
                    if pattern.search(subject):
                        pattern_field_counts[pattern_name] += 1

            messages = record.get("messages")
            if not isinstance(messages, list):
                messages = []
            if not messages:
                cases_without_messages += 1

            case_has_internal_note = False
            for message_index, message in enumerate(messages):
                if not isinstance(message, dict):
                    continue

                total_messages += 1
                role = message.get("role")
                if not isinstance(role, str) or not role:
                    role = "missing_or_unknown"
                message_roles[role] += 1

                if role == "internal_note" or truthy_note(message.get("note")):
                    case_has_internal_note = True

                content = message.get("content")
                content_html = message.get("content_html")
                if not (
                    isinstance(content, str) and content.strip()
                ) and not (
                    isinstance(content_html, str) and content_html.strip()
                ):
                    messages_missing_content += 1

                if not message.get("created_at") and not message.get("sent_at"):
                    messages_missing_created_at += 1

                attachments = message.get("attachments")
                if isinstance(attachments, list):
                    if attachments:
                        messages_with_attachments += 1
                    total_attachments += len(attachments)
                elif isinstance(attachments, dict) and attachments:
                    messages_with_attachments += 1
                    total_attachments += 1

                for text_index, text in enumerate(message_texts(message)):
                    fields_checked += 1
                    for pattern_name, pattern in PATTERNS.items():
                        if pattern.search(text):
                            pattern_field_counts[pattern_name] += 1
                            messages_with_suspected_patterns.add(
                                (case_id, f"{message_index}:{pattern_name}")
                            )

            if case_has_internal_note:
                cases_with_internal_notes += 1

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(DATASET_PATH),
        "jsonl_nonempty_lines": total_lines,
        "valid_json_records": valid_records,
        "malformed_lines": malformed_lines,
        "records_missing_case_id": missing_case_id_records,
        "unique_case_ids": len(seen_case_ids),
        "duplicate_case_id_count": len(duplicate_case_ids),
        "cases_without_messages": cases_without_messages,
        "cases_with_internal_notes": cases_with_internal_notes,
        "total_messages": total_messages,
        "message_roles": dict(sorted(message_roles.items())),
        "messages_missing_text": messages_missing_content,
        "messages_missing_timestamps": messages_missing_created_at,
        "messages_with_attachments": messages_with_attachments,
        "attachments_total": total_attachments,
        "text_fields_checked_for_patterns": fields_checked,
        "fields_matching_heuristics": dict(sorted(pattern_field_counts.items())),
        "note": (
            "Pattern checks are approximate and can produce false positives or miss PII. "
            "No matched text or personal data values are written to this report."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nСводный отчёт сохранён: {REPORT_PATH}")
    print("Исходный dataset.jsonl не изменялся.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
