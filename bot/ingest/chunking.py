"""Нарезка документов на чанки по абзацам.

- Короткий документ — один чанк.
- Длинный режется по абзацам до `max_chars`; соседние чанки перекрываются хвостом предыдущего
  (до `overlap_chars`), чтобы ответ на стыке не потерялся.
- Абзац длиннее `max_chars` режется по предложениям, а предложение — по символам.
- `embed_text` = заголовок документа + путь разделов + текст чанка: по нему строятся векторы.
  Пользователю и LLM уходит `text`, заголовок и источник передаются отдельно.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ingest.documents import Document

SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    url: str
    section: str
    text: str
    embed_text: str
    title_text: str            # заголовок + путь разделов: для вектора BM25 по заголовкам
    index: int
    meta: dict[str, Any] = field(default_factory=dict)


def _pieces(text: str, max_chars: int) -> list[str]:
    out: list[str] = []
    for para in (p.strip() for p in text.split("\n")):
        if not para:
            continue
        if len(para) <= max_chars:
            out.append(para)
            continue
        cur = ""
        for sent in SENTENCE_RE.split(para):
            while len(sent) > max_chars:
                if cur:
                    out.append(cur)
                    cur = ""
                out.append(sent[:max_chars])
                sent = sent[max_chars:]
            if cur and len(cur) + 1 + len(sent) > max_chars:
                out.append(cur)
                cur = sent
            else:
                cur = f"{cur} {sent}".strip()
        if cur:
            out.append(cur)
    return out


def split_text(text: str, max_chars: int = 900, overlap_chars: int = 150) -> list[str]:
    pieces = _pieces(text, max_chars)
    chunks: list[str] = []
    cur: list[str] = []
    for piece in pieces:
        if cur and sum(len(p) + 1 for p in cur) + len(piece) > max_chars:
            chunks.append("\n".join(cur))
            tail: list[str] = []
            for p in reversed(cur):              # перекрытие: последние абзацы предыдущего чанка
                if sum(len(t) + 1 for t in tail) + len(p) > overlap_chars:
                    break
                tail.insert(0, p)
            cur = tail
        cur.append(piece)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def chunk_document(doc: Document, max_chars: int = 900, overlap_chars: int = 150) -> list[ChunkRecord]:
    section = " → ".join(p for p in doc.section_path if p)
    header = doc.title if not section else f"{section}\n{doc.title}" if doc.source != "manual" else section
    records = []
    for i, text in enumerate(split_text(doc.text, max_chars, overlap_chars)):
        records.append(ChunkRecord(
            chunk_id=f"{doc.doc_id}#{i}",
            doc_id=doc.doc_id,
            source=doc.source,
            title=doc.title,
            url=doc.url,
            section=section,
            text=text,
            embed_text=f"{header}\n{text}",
            title_text=header,
            index=i,
            meta={k: v for k, v in doc.meta.items() if v is not None},
        ))
    return records
