"""Загрузчики, чанкинг, BM25 и поиск по временному локальному Qdrant (без модели эмбеддингов)."""
import asyncio
import json
import math
import zipfile

import pytest

from ingest.chunking import chunk_document, split_text
from ingest.documents import Document, clean_message, load_manual, load_omnidesk
from polybot.clients.qdrant_retriever import QdrantRetriever
from polybot.errors import ToolError
from polybot.retrieval.encoders import SparseBM25, lemmas, term_index
from polybot.retrieval.store import DENSE, SPARSE, connect, create_collection, manifest_path, point_id
from polybot.settings import RetrievalSettings


# ---------------------------------------------------------------- чанкинг
def test_short_text_is_one_chunk():
    assert split_text("Короткий абзац.", 900, 150) == ["Короткий абзац."]


def test_long_text_split_with_overlap():
    paras = [f"Абзац номер {i}. " + "слово " * 30 for i in range(10)]
    chunks = split_text("\n".join(paras), max_chars=500, overlap_chars=250)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    # хвост предыдущего чанка повторяется в начале следующего
    assert chunks[0].splitlines()[-1] == chunks[1].splitlines()[0]


def test_very_long_paragraph_is_cut():
    chunks = split_text("а" * 2000, max_chars=500, overlap_chars=0)
    assert [len(c) for c in chunks] == [500, 500, 500, 500]


def test_chunk_ids_and_header():
    doc = Document(doc_id="faq:x", source="faq", title="Заголовок", text="Текст.", url="u",
                   section_path=["Раздел"])
    [c] = chunk_document(doc)
    assert c.chunk_id == "faq:x#0" and c.embed_text.startswith("Раздел\nЗаголовок\n")


# ---------------------------------------------------------------- OmniDesk
def test_quotes_and_signatures_removed():
    msg = ("Как настроить формат? Четверг, 8 октября 2026, 20:49 +05:00 от Центр поддержки <[EMAIL]>: "
           "Добрый день! Старый ответ.")
    assert clean_message(msg) == "Как настроить формат?"
    assert clean_message("Ответ по существу.\nС уважением, Служба поддержки [PHONE]") == "Ответ по существу."


def test_omnidesk_case_without_support_reply_skipped(tmp_path):
    rows = [
        {"source_case_id": 1, "subject": "Есть ответ", "messages": [
            {"role": "customer", "text": "Вопрос?"},
            {"role": "support", "text": "Подробный ответ поддержки по существу вопроса."}]},
        {"source_case_id": 2, "subject": "Нет ответа", "messages": [{"role": "customer", "text": "?"}]},
    ]
    (tmp_path / "omnidesk_cases.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    docs = load_omnidesk(tmp_path)
    assert [d.doc_id for d in docs] == ["omnidesk:1"]
    assert docs[0].text.startswith("Клиент: Вопрос?\n\nПоддержка: Подробный")


# ---------------------------------------------------------------- руководство (.docx)
STYLES = """<?xml version="1.0"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:styleId="10"><w:name w:val="Згл 1 многоур"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>
<w:style w:styleId="11"><w:name w:val="heading 1"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>
<w:style w:styleId="24"><w:name w:val="heading 2"/><w:pPr><w:outlineLvl w:val="1"/></w:pPr></w:style>
<w:style w:styleId="20"><w:name w:val="Згл 2 многоур"/><w:basedOn w:val="24"/></w:style>
<w:style w:styleId="15"><w:name w:val="toc 1"/></w:style>
</w:styles>"""


def _p(text, style=None):
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}<w:r><w:t>{text}</w:t></w:r></w:p>"


def test_manual_sections_numbered(tmp_path):
    body = "".join([
        _p("Глава 1 5", "15"),                  # оглавление — пропускается
        _p("ВВЕДЕНИЕ", "11"), _p("О книге."),
        _p("", "10"), _p("Справочники", "11"), _p("Общий текст главы."),
        _p("Справочник Оборудование", "20"), _p("Текст про оборудование."),
        _p("Справочник Формат", "20"), _p("Текст про формат."),
    ])
    doc = ('<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/'
           f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>')
    path = tmp_path / "m.docx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/styles.xml", STYLES)
        z.writestr("word/document.xml", doc)
    docs = load_manual(path)
    assert [d.doc_id for d in docs] == ["manual:s0", "manual:1", "manual:1.1", "manual:1.2"]
    assert docs[2].section_path == ["Глава 1. Справочники", "Справочник Оборудование"]
    assert docs[3].text == "Текст про формат."
    assert "раздел 1.2" in docs[3].url


# ---------------------------------------------------------------- BM25
def test_lemmas_normalize_forms():
    assert lemmas("Расчёта заказов в программе") == lemmas("расчёт заказ программа")


def test_bm25_vectors():
    bm = SparseBM25(avgdl=5)
    idx, val = bm.doc_vector("заказ заказ бумага")
    assert idx == sorted(idx) and len(idx) == 2
    assert val[idx.index(term_index("заказ"))] > val[idx.index(term_index("бумага"))]
    q_idx, q_val = SparseBM25.query_vector("заказы")
    assert q_idx == [term_index("заказ")] and q_val == [1.0]


# ---------------------------------------------------------------- поиск по локальному Qdrant
class BagEncoder:
    """Детерминированный «эмбеддер» для теста: мешок лемм в 64 измерениях."""

    model_name = "bag-test"
    dim = 64

    def _vec(self, text):
        v = [0.0] * self.dim
        for t in lemmas(text):
            v[term_index(t) % self.dim] += 1
        n = math.sqrt(sum(x * x for x in v)) or 1
        return [x / n for x in v]

    def passages(self, texts):
        return [self._vec(t) for t in texts]

    def query(self, text):
        return self._vec(text)


@pytest.fixture
def local_index(tmp_path):
    from qdrant_client import models

    cfg = RetrievalSettings(qdrant_path=tmp_path / "q", collection="test_kb", dense_model="bag-test",
                            prefetch_k=10)
    docs = [Document("faq:1", "faq", "Замена материала", "Как учесть замену материала в заказе."),
            Document("faq:2", "faq", "Лицензии", "Сколько лицензий нужно на пользователей."),
            Document("omnidesk:7", "omnidesk", "Замена бумаги", "Поддержка: замену материала делают так.")]
    chunks = [c for d in docs for c in chunk_document(d)]
    enc, bm = BagEncoder(), SparseBM25(avgdl=8)
    client = connect(cfg, async_client=False)
    create_collection(client, cfg.collection, enc.dim)
    client.upsert(cfg.collection, points=[models.PointStruct(
        id=point_id(c.chunk_id),
        vector={DENSE: v, SPARSE: models.SparseVector(indices=bm.doc_vector(c.embed_text)[0],
                                                      values=bm.doc_vector(c.embed_text)[1])},
        payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "source": c.source, "title": c.title,
                 "url": c.url, "section": c.section, "text": c.text})
        for c, v in zip(chunks, enc.passages([c.embed_text for c in chunks]))])
    client.close()
    manifest_path(cfg).write_text(json.dumps({"version": "test", "dense_model": "bag-test"}))
    return cfg, enc


def test_hybrid_search_ranks_and_scores(local_index):
    cfg, enc = local_index
    r = QdrantRetriever(cfg, encoder=enc)
    chunks = asyncio.run(r.search("замена материала в заказе", 3))
    assert chunks[0].meta["doc_id"] == "faq:1"
    assert chunks[0].meta["rank"] == 1 and 0 < chunks[0].score <= 1.0001
    assert r.version == "test_kb@test"


def test_exclude_doc_ids(local_index):
    cfg, enc = local_index
    r = QdrantRetriever(cfg, encoder=enc)
    chunks = asyncio.run(r.search("замена материала", 3, exclude_doc_ids=["omnidesk:7"]))
    assert "omnidesk:7" not in {c.meta["doc_id"] for c in chunks}


def test_omnidesk_quota_puts_official_first(local_index):
    cfg, enc = local_index
    r = QdrantRetriever(cfg.model_copy(update={"omnidesk_k": 1}), encoder=enc)
    chunks = asyncio.run(r.search("замена материала", 3))
    sources = [c.meta["source"] for c in chunks]
    assert sources.count("omnidesk") == 1 and sources[-1] == "omnidesk"
    assert [c.meta["rank"] for c in chunks] == [1, 2, 3]


def test_dense_and_bm25_modes(local_index):
    cfg, enc = local_index
    async def run(mode):
        r = QdrantRetriever(cfg.model_copy(update={"mode": mode, "omnidesk_k": 5}), encoder=enc)
        try:
            return await r.search("замена материала в заказе", 2)
        finally:
            await r.close()     # локальный Qdrant открывается только одним клиентом

    for mode in ("dense", "bm25"):
        assert asyncio.run(run(mode))[0].meta["doc_id"] in ("faq:1", "omnidesk:7")


def test_title_signal_finds_section_by_name(tmp_path):
    """Вопрос называет раздел; текст раздела общий — находит BM25 по заголовкам."""
    from qdrant_client import models

    from polybot.retrieval.store import TITLE

    cfg = RetrievalSettings(qdrant_path=tmp_path / "q", collection="t_kb", dense_model="bag-test",
                            title_weight=3.0, bm25_weight=1.0, omnidesk_k=0, prefetch_k=10)
    docs = [Document("manual:5.31", "manual", "Заказы готовые к планированию",
                     "Отчёт показывает список и позволяет отобрать строки.",
                     section_path=["Глава 5. Отчеты", "Заказы готовые к планированию"]),
            Document("manual:1.1", "manual", "Справочник Оборудование",
                     "Заказы планирование заказы планирование заказы.",
                     section_path=["Глава 1. Справочники", "Справочник Оборудование"])]
    chunks = [c for d in docs for c in chunk_document(d)]
    enc, bm, tbm = BagEncoder(), SparseBM25(avgdl=8), SparseBM25(avgdl=6)
    client = connect(cfg, async_client=False)
    create_collection(client, cfg.collection, enc.dim)
    pts = []
    for c in chunks:
        i, v = bm.doc_vector(c.embed_text)
        ti, tv = tbm.doc_vector(c.title_text)
        pts.append(models.PointStruct(
            id=point_id(c.chunk_id),
            vector={DENSE: enc.query(c.text), SPARSE: models.SparseVector(indices=i, values=v),
                    TITLE: models.SparseVector(indices=ti, values=tv)},
            payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "source": c.source,
                     "title": c.title, "url": "", "section": c.section, "text": c.text}))
    client.upsert(cfg.collection, points=pts)
    client.close()
    manifest_path(cfg).write_text(json.dumps({"version": "t", "dense_model": "bag-test",
                                              "vectors": [DENSE, SPARSE, TITLE]}))

    async def top(weight):
        r = QdrantRetriever(cfg.model_copy(update={"title_weight": weight}), encoder=enc)
        try:
            return (await r.search("Отчёт «Заказы готовые к планированию» — как отобрать?", 2))[0]
        finally:
            await r.close()

    assert asyncio.run(top(3.0)).meta["doc_id"] == "manual:5.31"


def test_missing_collection_is_tool_error(local_index):
    cfg, enc = local_index
    r = QdrantRetriever(cfg.model_copy(update={"collection": "test_kb"}), encoder=enc)
    r.cfg = cfg.model_copy(update={"collection": "nope", "max_retries": 0})
    with pytest.raises(ToolError) as exc:
        asyncio.run(r.search("замена", 3))
    assert exc.value.tool == "retrieve" and exc.value.kind == "unavailable"


def test_missing_index_explained(tmp_path):
    cfg = RetrievalSettings(qdrant_path=tmp_path / "q", collection="absent")
    with pytest.raises(RuntimeError, match="build_index"):
        QdrantRetriever(cfg, encoder=BagEncoder())
