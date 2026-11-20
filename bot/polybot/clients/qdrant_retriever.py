"""Гибридный поиск по базе знаний в Qdrant: dense (fastembed) + BM25 по леммам, взвешенный RRF.

Контракт инструмента `retrieve` (см. clients/retriever.py):
- `await search(query, top_k, exclude_doc_ids=()) -> list[Chunk]`;
- таймаут `timeout_s` на попытку, `max_retries` повторов; при сбое — `ToolError`;
- `Chunk.score` — косинус между запросом и чанком (по нему работает evidence_gate);
  ранг и оценка RRF — в `Chunk.meta`.

`exclude_doc_ids` нужен харнессу: кейс dev, сделанный из обращения OmniDesk, не должен
находить само это обращение.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from polybot.errors import ToolError, ToolErrorKind
from polybot.retrieval.encoders import DenseEncoder, SparseBM25
from polybot.retrieval.store import DENSE, SPARSE, TITLE, connect, read_manifest
from polybot.schemas import Chunk
from polybot.settings import RetrievalSettings

logger = logging.getLogger(__name__)
RRF_K = 60


class QdrantRetriever:
    def __init__(self, cfg: RetrievalSettings, encoder: DenseEncoder | None = None):
        self.cfg = cfg
        self.manifest = read_manifest(cfg)
        if not self.manifest:
            raise RuntimeError(
                f"нет манифеста индекса {cfg.collection}: соберите его — python -m ingest.build_index")
        if self.manifest.get("dense_model") != cfg.dense_model:
            raise RuntimeError(f"индекс собран моделью {self.manifest.get('dense_model')}, "
                               f"а в настройках {cfg.dense_model}: пересоберите индекс")
        self.version = f"{cfg.collection}@{self.manifest['version']}"
        self.has_title = TITLE in self.manifest.get("vectors", [])
        self.encoder = encoder or DenseEncoder(cfg.dense_model, cfg.models_dir)
        self._client = connect(cfg, async_client=True)

    async def close(self) -> None:
        await self._client.close()

    async def search(self, query: str, top_k: int,
                     exclude_doc_ids: Iterable[str] = ()) -> list[Chunk]:
        exclude = list(exclude_doc_ids)
        last: ToolError | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                return await asyncio.wait_for(self._search(query, top_k, exclude),
                                              timeout=self.cfg.timeout_s)
            except TimeoutError as exc:
                last = ToolError("retrieve", ToolErrorKind.TIMEOUT, f"попытка {attempt + 1}")
                last.__cause__ = exc
            except ToolError as exc:
                last = exc
            except Exception as exc:  # noqa: BLE001 — сетевые ошибки клиента Qdrant
                last = ToolError("retrieve", ToolErrorKind.UNAVAILABLE, f"{type(exc).__name__}: {exc}")
                last.__cause__ = exc
            logger.warning("retrieve: %s", last)
        assert last is not None
        raise last

    async def _search(self, query: str, top_k: int, exclude: list[str]) -> list[Chunk]:
        """Официальные источники (FAQ, руководство) и архив OmniDesk ищутся раздельно: в архиве
        втрое больше чанков, и без квоты он вытесняет FAQ. В выдаче сначала официальные,
        затем не более `omnidesk_k` чанков архива (C03: приоритет официальных материалов)."""
        dense = await asyncio.to_thread(self.encoder.query, query)
        omni_k = min(self.cfg.omnidesk_k, top_k)
        if omni_k >= top_k:                       # квота выключена: одна общая выдача
            points = await self._query(dense, query, top_k, exclude, None)
        else:
            official = await self._query(dense, query, top_k - omni_k, exclude, ["faq", "manual"])
            omni = await self._query(dense, query, omni_k, exclude, ["omnidesk"]) if omni_k else []
            points = official + omni
        chunks = []
        for rank, p in enumerate(points, 1):
            vec = p.vector[DENSE] if isinstance(p.vector, dict) else p.vector
            cosine = sum(a * b for a, b in zip(dense, vec))
            pl = p.payload or {}
            chunks.append(Chunk(
                chunk_id=pl["chunk_id"], text=pl["text"], title=pl.get("title", ""),
                url=pl.get("url", ""), score=round(float(cosine), 4),
                meta={"doc_id": pl["doc_id"], "source": pl.get("source"),
                      "section": pl.get("section", ""), "rank": rank, "rrf": round(p.score, 5)},
            ))
        return chunks

    async def _query(self, dense: list[float], query: str, limit: int, exclude: list[str],
                     sources: list[str] | None):
        from qdrant_client import models

        must = [models.FieldCondition(key="source", match=models.MatchAny(any=sources))] if sources else []
        must_not = [models.FieldCondition(key="doc_id", match=models.MatchAny(any=exclude))] if exclude else []
        flt = models.Filter(must=must, must_not=must_not) if (must or must_not) else None
        mode = self.cfg.mode
        idx, val = SparseBM25.query_vector(query)
        async def dense_points(n):
            r = await self._client.query_points(self.cfg.collection, query=dense, using=DENSE,
                                                query_filter=flt, limit=n, with_payload=True,
                                                with_vectors=[DENSE])
            return r.points

        async def sparse_points(n, using=SPARSE):
            if not idx:
                return []
            r = await self._client.query_points(
                self.cfg.collection, query=models.SparseVector(indices=idx, values=val), using=using,
                query_filter=flt, limit=n, with_payload=True, with_vectors=[DENSE])
            return r.points

        if mode == "dense":
            return await dense_points(limit)
        if mode == "bm25":
            return await sparse_points(limit)
        # Гибрид: взвешенный RRF на стороне клиента из трёх списков — dense, BM25 по тексту
        # и BM25 по заголовкам разделов (на dev BM25 лучше находит руководство, dense — перефразы FAQ).
        lists = [(1.0, await dense_points(self.cfg.prefetch_k)),
                 (self.cfg.bm25_weight, await sparse_points(self.cfg.prefetch_k))]
        if self.cfg.title_weight > 0 and self.has_title:
            lists.append((self.cfg.title_weight, await sparse_points(self.cfg.prefetch_k, TITLE)))
        fused: dict = {}
        for weight, pts in lists:
            for rank, pt in enumerate(pts, 1):
                score, _ = fused.get(pt.id, (0.0, pt))
                fused[pt.id] = (score + weight / (RRF_K + rank), pt)
        ranked = sorted(fused.values(), key=lambda x: x[0], reverse=True)[:limit]
        for score, pt in ranked:
            pt.score = score
        return [pt for _, pt in ranked]
