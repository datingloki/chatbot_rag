"""Схема коллекции Qdrant и подключение — общее для индексации и поиска."""
from __future__ import annotations

import json
import uuid
import warnings
from pathlib import Path
from typing import Any

from polybot.settings import BOT_DIR, RetrievalSettings

DENSE = "dense"
SPARSE = "bm25"
TITLE = "title_bm25"      # BM25 по заголовку документа и пути разделов


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"polybot:{chunk_id}"))


def connect(cfg: RetrievalSettings, *, async_client: bool):
    from qdrant_client import AsyncQdrantClient, QdrantClient

    cls = AsyncQdrantClient if async_client else QdrantClient
    if cfg.qdrant_path is not None:
        return cls(path=str(cfg.qdrant_path))
    return cls(url=cfg.qdrant_url, timeout=int(cfg.timeout_s) or 1)


def manifest_path(cfg: RetrievalSettings) -> Path:
    """Манифест индекса лежит рядом с данными: для локального Qdrant — рядом с его папкой
    (по умолчанию bot/data/<коллекция>.manifest.json), для сервера — в bot/data/."""
    base = Path(cfg.qdrant_path).parent if cfg.qdrant_path is not None else BOT_DIR / "data"
    return base / f"{cfg.collection}.manifest.json"


def read_manifest(cfg: RetrievalSettings) -> dict[str, Any]:
    p = manifest_path(cfg)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def create_collection(client, name: str, dim: int) -> None:
    from qdrant_client import models

    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF),
                               TITLE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    # Индекс по doc_id ускоряет фильтр exclude_doc_ids на сервере; локальный Qdrant его
    # не поддерживает и только предупреждает об этом.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client.create_payload_index(name, "doc_id", models.PayloadSchemaType.KEYWORD)
