"""Сборка индекса базы знаний в Qdrant.

    .venv/bin/python -m ingest.build_index                 # все источники
    .venv/bin/python -m ingest.build_index --sources faq   # только FAQ (быстро, для проверки)

Коллекция пересоздаётся целиком. Рядом пишется манифест `bot/data/<коллекция>.manifest.json`:
версия индекса (хеш источников и параметров), модель эмбеддингов, параметры чанкинга, число
документов и чанков по источникам. Версия индекса попадает в метаданные прогонов харнесса.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter

from ingest.chunking import chunk_document
from ingest.documents import FAQ_FILES, KB_DIR, MANUAL_FILE, OMNIDESK_FILE, load_all
from polybot.retrieval.encoders import DenseEncoder, SparseBM25, average_length
from polybot.retrieval.store import DENSE, SPARSE, TITLE, connect, create_collection, manifest_path, point_id
from polybot.settings import Settings

SOURCE_FILES = {"faq": FAQ_FILES, "manual": (MANUAL_FILE,), "omnidesk": (OMNIDESK_FILE,)}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="faq,manual,omnidesk")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())

    cfg = Settings().retrieval
    t0 = time.time()
    docs = load_all(KB_DIR, sources)
    chunks = [c for d in docs for c in chunk_document(d, cfg.chunk_chars, cfg.chunk_overlap)]
    print(f"документов {len(docs)}, чанков {len(chunks)} ({dict(Counter(c.source for c in chunks))}) "
          f"за {time.time() - t0:.1f} с")

    files = {name: sha256_file(KB_DIR / name)[:16] for s in sources for name in SOURCE_FILES[s]}
    params = {"schema": 2, "dense_model": cfg.dense_model, "chunk_chars": cfg.chunk_chars,
              "chunk_overlap": cfg.chunk_overlap, "sources": list(sources), "files": files}
    if cfg.dense_max_tokens != 128:                  # по умолчанию — прежняя версия индекса
        params["dense_max_tokens"] = cfg.dense_max_tokens
    version = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]

    t1 = time.time()
    avgdl = average_length([c.embed_text for c in chunks])
    sparse = SparseBM25(avgdl)
    title_avgdl = average_length([c.title_text for c in chunks])
    title_sparse = SparseBM25(title_avgdl)
    encoder = DenseEncoder(cfg.dense_model, cfg.models_dir, cfg.dense_max_tokens)
    print(f"модель {cfg.dense_model} (dim {encoder.dim}) загружена за {time.time() - t1:.1f} с")

    from qdrant_client import models

    cfg.qdrant_path and cfg.qdrant_path.mkdir(parents=True, exist_ok=True)
    client = connect(cfg, async_client=False)
    create_collection(client, cfg.collection, encoder.dim)
    t2 = time.time()
    for start in range(0, len(chunks), args.batch):
        batch = chunks[start:start + args.batch]
        dense = encoder.passages([c.embed_text for c in batch])
        points = []
        for c, vec in zip(batch, dense):
            idx, val = sparse.doc_vector(c.embed_text)
            t_idx, t_val = title_sparse.doc_vector(c.title_text)
            points.append(models.PointStruct(
                id=point_id(c.chunk_id),
                vector={DENSE: vec, SPARSE: models.SparseVector(indices=idx, values=val),
                        TITLE: models.SparseVector(indices=t_idx, values=t_val)},
                payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "source": c.source,
                         "title": c.title, "url": c.url, "section": c.section, "text": c.text,
                         "index": c.index, **{f"meta_{k}": v for k, v in c.meta.items()}},
            ))
        client.upsert(cfg.collection, points=points)
        done = start + len(batch)
        if done % (args.batch * 10) == 0 or done == len(chunks):
            rate = done / max(time.time() - t2, 1e-6)
            print(f"  {done}/{len(chunks)} чанков, {rate:.1f}/с")
    client.close()

    manifest = {"version": version, "collection": cfg.collection, **params, "avgdl": avgdl,
                "title_avgdl": title_avgdl, "vectors": [DENSE, SPARSE, TITLE],
                "dim": encoder.dim, "documents": dict(Counter(d.source for d in docs)),
                "chunks": dict(Counter(c.source for c in chunks)), "total_chunks": len(chunks),
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "build_seconds": round(time.time() - t0, 1)}
    path = manifest_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"индекс {cfg.collection}@{version}: {len(chunks)} чанков за {time.time() - t0:.0f} с → {path}")


if __name__ == "__main__":
    main()
