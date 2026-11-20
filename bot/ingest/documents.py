"""Загрузчики базы знаний: каждый источник → список `Document` одного вида.

| Источник | Файл | Документ |
|---|---|---|
| FAQ и лицензии | `knowledge base/faq_articles.jsonl`, `faq_license_install_articles.jsonl` | статья |
| Руководство | `knowledge base/manual_unf8_poly2_3.0.docx` | раздел (по стилям заголовков Word) |
| OmniDesk | `knowledge base/omnidesk_cases.jsonl` (персональные данные уже замаскированы) | обращение |
"""
from __future__ import annotations

import ast
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

KB_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge base"
FAQ_FILES = ("faq_articles.jsonl", "faq_license_install_articles.jsonl")
MANUAL_FILE = "manual_unf8_poly2_3.0.docx"
OMNIDESK_FILE = "omnidesk_cases.jsonl"

MANUAL_PRODUCT = "1С:УНФ 8. Полиграфия 2"
MANUAL_RELEASE = "3.0"


@dataclass
class Document:
    doc_id: str                 # faq:poly2-261 | manual:6.2.1 | omnidesk:261420480
    source: str                 # faq | manual | omnidesk
    title: str
    text: str
    url: str = ""
    section_path: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _jsonish(value: Any) -> Any:
    """Поля в jsonl FAQ сохранены как repr словаря Python: "{'product': ...}"."""
    if isinstance(value, str) and value[:1] in "{[":
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


# ---------------------------------------------------------------- FAQ
def load_faq(kb_dir: Path = KB_DIR) -> list[Document]:
    docs = []
    for name in FAQ_FILES:
        for line in (kb_dir / name).open(encoding="utf-8"):
            if not line.strip():
                continue
            a = json.loads(line)
            scope = _jsonish(a.get("target_scope")) or {}
            section = _jsonish(a.get("source_section")) or {}
            docs.append(Document(
                doc_id=f"faq:{a['id']}",
                source="faq",
                title=a["title"].strip(),
                text=a["text"].strip(),
                url=a.get("source_url", ""),
                section_path=[section.get("title", "")] if isinstance(section, dict) else [],
                meta={"product": scope.get("product") if isinstance(scope, dict) else None,
                      "release": scope.get("release") if isinstance(scope, dict) else None,
                      "file": name, "content_sha256": a.get("content_sha256")},
            ))
    return docs


# ---------------------------------------------------------------- руководство (.docx)
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _styles(z: zipfile.ZipFile) -> dict[str, dict[str, Any]]:
    root = ET.fromstring(z.read("word/styles.xml"))
    out = {}
    for s in root.iter(f"{W}style"):
        sid = s.get(f"{W}styleId")
        name = s.find(f"{W}name")
        based = s.find(f"{W}basedOn")
        lvl = s.find(f"{W}pPr/{W}outlineLvl")
        out[sid] = {"name": name.get(f"{W}val") if name is not None else "",
                    "based": based.get(f"{W}val") if based is not None else None,
                    "lvl": int(lvl.get(f"{W}val")) if lvl is not None else None}
    return out


def _heading_level(sid: str | None, styles: dict[str, dict[str, Any]]) -> int | None:
    """Уровень заголовка 1..9 по outlineLvl стиля (с наследованием basedOn)."""
    seen = set()
    while sid and sid not in seen and sid in styles:
        seen.add(sid)
        st = styles[sid]
        if st["name"].lower().startswith("toc"):
            return None
        if st["lvl"] is not None and st["lvl"] < 9:
            return st["lvl"] + 1
        sid = st["based"]
    return None


def _para_text(p: ET.Element) -> str:
    parts = []
    for node in p.iter():
        if node.tag == f"{W}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{W}tab":
            parts.append(" ")
        elif node.tag in (f"{W}br", f"{W}cr"):
            parts.append("\n")
    return re.sub(r"[  ]+", " ", "".join(parts)).strip()


def _table_text(tbl: ET.Element) -> str:
    rows = []
    for tr in tbl.iter(f"{W}tr"):
        cells = [" ".join(_para_text(p) for p in tc.iter(f"{W}p")).strip() for tc in tr.iter(f"{W}tc")]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def load_manual(path: Path | None = None) -> list[Document]:
    """Раздел руководства = заголовок любого уровня + текст до следующего заголовка.

    Номер главы в документе ставит автонумерация Word (пустой абзац стиля «Згл 1 многоур»
    перед заголовком главы), поэтому номера глав и подразделов считаем сами: 6, 6.2, 6.2.1.
    Оглавление (стили toc) пропускаем.
    """
    path = path or KB_DIR / MANUAL_FILE
    with zipfile.ZipFile(path) as z:
        styles = _styles(z)
        body = ET.fromstring(z.read("word/document.xml")).find(f"{W}body")

    numbering_marker = {sid for sid, st in styles.items() if st["name"].startswith("Згл 1")}
    docs: list[Document] = []
    path_titles: list[str] = []
    counters = [0] * 9
    chapter_numbered = False
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(t for t in buf if t).strip()
        buf.clear()
        if not path_titles or not text or path_titles[0] == "Оглавление":
            return
        number = ".".join(str(c) for c in counters[:len(path_titles)]) if chapter_numbered else ""
        docs.append(Document(
            doc_id=f"manual:{number or 's' + str(len(docs))}",
            source="manual",
            title=path_titles[-1],
            text=text,
            url=f"Руководство «УНФ 8. Полиграфия 2», ред. {MANUAL_RELEASE}"
                + (f", раздел {number}" if number else f", «{path_titles[0]}»"),
            section_path=[(f"Глава {counters[0]}. " if chapter_numbered and i == 0 else "") + t
                          for i, t in enumerate(path_titles)],
            meta={"product": MANUAL_PRODUCT, "release": MANUAL_RELEASE, "number": number,
                  "chapter": counters[0] if chapter_numbered else None},
        ))

    pending_chapter = False
    for el in body:
        if el.tag == f"{W}tbl":
            buf.append(_table_text(el))
            continue
        if el.tag != f"{W}p":
            continue
        ps = el.find(f"{W}pPr/{W}pStyle")
        sid = ps.get(f"{W}val") if ps is not None else None
        text = _para_text(el)
        if sid in numbering_marker:
            pending_chapter = True          # следующий заголовок 1-го уровня — нумерованная глава
            continue
        level = _heading_level(sid, styles)
        if level is not None and text:
            flush()
            if level == 1:
                chapter_numbered = pending_chapter
                if pending_chapter:
                    counters[0] += 1
                pending_chapter = False
                counters[1:] = [0] * 8
            else:
                counters[level - 1] += 1
                counters[level:] = [0] * (9 - level)
            path_titles[:] = path_titles[:level - 1] + [re.sub(r"\s+", " ", text)]
            continue
        if text:
            buf.append(text)
    flush()
    return docs


# ---------------------------------------------------------------- OmniDesk
# Цитаты предыдущих писем и подписи — шум для поиска.
QUOTE_RE = re.compile(
    r"(От кого:|From:|-{5,}\s*(Original|Исходное)|Отправлено с|\d{1,2}[./]\d{1,2}[./]\d{2,4}[^\n]{0,60}пишет:"
    r"|пишет:\s*$|Полезные материалы технической поддержки"
    r"|(Понедельник|Вторник|Среда|Четверг|Пятница|Суббота|Воскресенье),\s*\d{1,2}\s+\w+\s+\d{4}"
    r"|\bот\b[^\n<]{0,80}<\s*\[EMAIL\]\s*>\s*:)",
    re.I | re.M,
)
SIGNATURE_RE = re.compile(r"\n?\s*(С уважением|--\s*\n|Best regards)[\s\S]*$|\s--\s*$", re.I)
MAX_MESSAGE_CHARS = 2000


def clean_message(text: str) -> str:
    m = QUOTE_RE.search(text)
    if m:
        text = text[:m.start()]
    text = SIGNATURE_RE.sub("", text)
    return re.sub(r"[ \t ]+", " ", text).strip()[:MAX_MESSAGE_CHARS]


def load_omnidesk(kb_dir: Path = KB_DIR) -> list[Document]:
    """Обращение → документ «Клиент: … / Поддержка: …». Без ответа поддержки — пропускаем:
    фактов в нём нет."""
    docs = []
    for line in (kb_dir / OMNIDESK_FILE).open(encoding="utf-8"):
        if not line.strip():
            continue
        c = json.loads(line)
        msgs = c["messages"] if isinstance(c["messages"], list) else json.loads(c["messages"])
        turns = []
        for m in msgs:
            text = clean_message(m.get("text", ""))
            if text:
                turns.append(("Поддержка" if m.get("role") == "support" else "Клиент", text))
        if not any(role == "Поддержка" and len(t) >= 30 for role, t in turns):
            continue
        case_id = str(c["source_case_id"])
        docs.append(Document(
            doc_id=f"omnidesk:{case_id}",
            source="omnidesk",
            title=(c.get("subject") or "").strip() or f"Обращение {case_id}",
            text="\n\n".join(f"{role}: {t}" for role, t in turns),
            url="",
            section_path=["Архив ответов службы поддержки"],
            meta={"case_id": case_id, "created_at": c.get("created_at")},
        ))
    return docs


def load_all(kb_dir: Path = KB_DIR, sources: tuple[str, ...] = ("faq", "manual", "omnidesk")) -> list[Document]:
    docs: list[Document] = []
    if "faq" in sources:
        docs += load_faq(kb_dir)
    if "manual" in sources:
        docs += load_manual(kb_dir / MANUAL_FILE)
    if "omnidesk" in sources:
        docs += load_omnidesk(kb_dir)
    return docs
