"""Полный текст записи корпуса по стабильному ID из `evidence[].source_id` кейса.

    .venv/bin/python -m harness.show_source manual:12.3
    .venv/bin/python -m harness.show_source faq:poly2-261 omnidesk:261420480

Выдержка в кейсе подтверждает отдельный тезис; применимость инструкции, условия и исключения
видны только в записи целиком — её и печатает команда. Тот же разбор корпуса, что строит
индекс бота (`ingest.documents.load_all`), поэтому ID совпадают с источниками в ответах бота.
"""
from __future__ import annotations

import json
import sys

from ingest.documents import KB_DIR, OMNIDESK_FILE, load_all


def raw_omnidesk(case_id: str) -> dict | None:
    """Обращение, которого нет в индексе (нет содержательного ответа поддержки), — из файла корпуса."""
    for line in (KB_DIR / OMNIDESK_FILE).open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if str(r.get("source_case_id")) == case_id:
                return r
    return None


def main(ids: list[str]) -> int:
    if not ids:
        print(__doc__)
        return 1
    docs = {d.doc_id: d for d in load_all()}
    missing = 0
    for i in ids:
        d = docs.get(i)
        if d is None and i.startswith("omnidesk:") and (raw := raw_omnidesk(i.split(":", 1)[1])):
            print(f"== {i} | {raw.get('subject', '')}  (в файле корпуса, в индекс не входит)")
            print(json.dumps(raw.get("messages", []), ensure_ascii=False, indent=1), "\n")
            continue
        if d is None:
            print(f"== {i}: не найдено в корпусе")
            missing += 1
            continue
        print(f"== {d.doc_id} | {d.title}")
        print(f"   {d.url}  | {' / '.join(d.section_path)}")
        print(f"   {d.meta}\n")
        print(d.text, "\n")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
