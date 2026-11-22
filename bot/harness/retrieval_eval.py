"""Оценка поиска на dev (eval не читается) и подбор порогов evidence_gate.

    .venv/bin/python -m harness.retrieval_eval [--top-k 10] [--note "..."]

Для каждого кейса dev поисковый запрос — сообщения пользователя из истории + текущее (как
`problem_messages` в боте). Эталон берётся из `evidence` кейса:
- `faq_id` → документ `faq:<id>`;
- `source_case_id` → документ `omnidesk:<id>`;
- `excerpt` (руководство) → чанк, где есть ≥ 75 % слов фрагмента.

Кейс из обращения OmniDesk (`origin = real_anonymized_candidate:omnidesk:<id>:…`) ищет с
исключённым обращением-источником: иначе поиск находит сам вопрос. Если эталон — это же
обращение, кейс в recall не входит (эталон недостижим), но идёт в калибровку порога.

Отчёт: harness/results/retrieval_<время>.json — метаданные (версия индекса, модель, хеш dev,
git), метрики по группам, таблица порогов и строки по кейсам.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from ingest.chunking import chunk_document
from ingest.documents import load_manual
from polybot.clients.qdrant_retriever import QdrantRetriever
from polybot.settings import BOT_DIR, REPO_DIR, Settings

DEV = REPO_DIR / "dev set" / "dev_300_balanced.jsonl"
RESULTS = BOT_DIR / "harness" / "results"
WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.I)
EXCERPT_COVERAGE = 0.75
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(11)]   # 0.30 … 0.80


def words(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text) if len(w) > 2}


def case_query(case: dict) -> str:
    inp = case["input"]
    prev = [h["text"] for h in inp.get("history", []) if h.get("role") == "user"]
    return "\n".join([*prev, inp["user_message"]])


def origin_case(case: dict) -> str | None:
    m = re.match(r"real_anonymized_candidate:omnidesk:(\d+)", case.get("origin", ""))
    return m.group(1) if m else None


def gold(case: dict) -> tuple[set[str], list[set[str]]]:
    docs: set[str] = set()
    excerpts: list[set[str]] = []
    for e in case.get("evidence") or []:
        if e.get("faq_id"):
            docs.add(f"faq:{e['faq_id']}")
        if e.get("source_case_id"):
            docs.add(f"omnidesk:{e['source_case_id']}")
        if e.get("excerpt"):
            w = words(e["excerpt"])
            if len(w) >= 4:
                excerpts.append(w)
    return docs, excerpts


def group(case: dict, docs: set[str], excerpts: list[set[str]], own: str | None) -> str:
    if case["origin_type"] == "official_faq_entry":
        return "faq_verbatim"
    if any(d.startswith("faq:") for d in docs):
        return "faq_paraphrase"
    if excerpts:
        return "manual"
    if any(d.startswith("omnidesk:") for d in docs):
        reachable = {d for d in docs if d != f"omnidesk:{own}"}
        return "omnidesk_other" if reachable else "omnidesk_self"
    return "no_gold"


def first_hit(chunks, docs: set[str], excerpts: list[set[str]]) -> int | None:
    for rank, c in enumerate(chunks, 1):
        if c.meta.get("doc_id") in docs:
            return rank
        cw = words(c.text)
        if any(len(e & cw) / len(e) >= EXCERPT_COVERAGE for e in excerpts):
            return rank
    return None


def excerpt_sections(excerpts: list[set[str]], manual_chunks) -> set[str]:
    """Разделы руководства, в чанках которых есть эталонный фрагмент: для метрики «нужный раздел
    найден» (sec@k) — LLM может ответить и по соседнему чанку того же раздела."""
    return {doc_id for doc_id, cw in manual_chunks
            if any(len(e & cw) / len(e) >= EXCERPT_COVERAGE for e in excerpts)}


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=REPO_DIR, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "—"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    settings = Settings()
    cfg = settings.retrieval
    assert "eval" not in DEV.name.lower()
    cases = [json.loads(line) for line in DEV.open(encoding="utf-8") if line.strip()]
    retriever = QdrantRetriever(cfg)
    manual_chunks = [(c.doc_id, words(c.text)) for d in load_manual()
                     for c in chunk_document(d, cfg.chunk_chars, cfg.chunk_overlap)]

    rows, latencies = [], []
    for case in cases:
        docs, excerpts = gold(case)
        own = origin_case(case)
        exclude = [f"omnidesk:{own}"] if own else []
        t0 = time.perf_counter()
        chunks = await retriever.search(case_query(case), args.top_k, exclude)
        latencies.append((time.perf_counter() - t0) * 1000)
        g = group(case, docs, excerpts, own)
        reachable_docs = docs - set(exclude)
        scored = g not in ("omnidesk_self", "no_gold")
        rank = first_hit(chunks, reachable_docs, excerpts) if scored else None
        sec_docs = reachable_docs | excerpt_sections(excerpts, manual_chunks)
        sec_rank = next((i for i, c in enumerate(chunks, 1) if c.meta.get("doc_id") in sec_docs),
                        None) if scored else None
        rows.append({"case_id": case["case_id"], "class": case["primary_class"],
                     "expected_action": case["expected_action"], "group": g, "rank": rank, "sec_rank": sec_rank,
                     "best_score": max((c.score for c in chunks), default=0.0),
                     "top": [c.chunk_id for c in chunks[:5]], "excluded": exclude})
    await retriever.close()

    # ---------------------------------------------------------------- recall / MRR по группам
    print(f"\nИндекс {retriever.version}, модель {cfg.dense_model}, top_k={args.top_k}, "
          f"prefetch={cfg.prefetch_k}, mode={cfg.mode}, bm25_weight={cfg.bm25_weight}, "
          f"title_weight={cfg.title_weight}, omnidesk_k={cfg.omnidesk_k}; кейсов dev: {len(rows)}")
    print(f"Поиск: P50 {statistics.median(latencies):.0f} мс, "
          f"P95 {statistics.quantiles(latencies, n=20)[18]:.0f} мс\n")
    metrics = {}
    print(f"{'группа':16s} {'n':>4s} {'hit@1':>12s} {'hit@5':>12s} {'hit@10':>12s} {'MRR@10':>7s}"
          f" {'раздел@5':>12s}")
    for g in ("faq_verbatim", "faq_paraphrase", "manual", "omnidesk_other", "omnidesk_self", "no_gold"):
        rs = [r for r in rows if r["group"] == g]
        if not rs:
            continue
        if g in ("omnidesk_self", "no_gold"):
            print(f"{g:16s} {len(rs):4d}   эталон недостижим/нет — только калибровка порога")
            metrics[g] = {"n": len(rs)}
            continue
        hit = {k: sum(1 for r in rs if r["rank"] and r["rank"] <= k) for k in (1, 5, 10)}
        mrr = sum(1 / r["rank"] for r in rs if r["rank"]) / len(rs)
        sec5 = sum(1 for r in rs if r["sec_rank"] and r["sec_rank"] <= 5)
        metrics[g] = {"n": len(rs), **{f"hit@{k}": v for k, v in hit.items()}, "mrr@10": round(mrr, 3),
                      "section@5": sec5}
        print(f"{g:16s} {len(rs):4d} {pct(hit[1], len(rs)):>12s} {pct(hit[5], len(rs)):>12s} "
              f"{pct(hit[10], len(rs)):>12s} {mrr:7.3f} {pct(sec5, len(rs)):>12s}")

    # ---------------------------------------------------------------- пороги evidence_gate
    buckets = defaultdict(list)
    for r in rows:
        if r["expected_action"] == "answer" and r["rank"] and r["rank"] <= 5:
            buckets["answer_hit@5"].append(r["best_score"])
        if r["class"] == "K4":
            buckets["K4_no_answer"].append(r["best_score"])
        if r["class"] == "K1":
            buckets["K1_out_of_domain"].append(r["best_score"])
        if r["class"] == "K6":
            buckets["K6_client_specific"].append(r["best_score"])
    print("\nДоля кейсов, у которых косинус лучшего чанка ≥ порога (проходят evidence_gate):")
    names = ["answer_hit@5", "K4_no_answer", "K1_out_of_domain", "K6_client_specific"]
    print("порог  " + "  ".join(f"{n:>20s}" for n in names))
    table = []
    for t in THRESHOLDS:
        line = {"threshold": t}
        cells = []
        for n in names:
            xs = buckets[n]
            k = sum(1 for x in xs if x >= t)
            line[n] = round(k / len(xs), 3) if xs else None
            cells.append(f"{pct(k, len(xs)):>20s}")
        table.append(line)
        print(f"{t:5.2f}  " + "  ".join(cells))
    best = max(table, key=lambda r: (r["answer_hit@5"] or 0) - (r["K4_no_answer"] or 0))
    print(f"\nМаксимум разницы «ответимые − K4»: порог {best['threshold']} "
          f"(ответимые {best['answer_hit@5']}, K4 {best['K4_no_answer']}). "
          f"Сейчас в настройках min_score={cfg.min_score}.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"retrieval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    meta = {"git": git_sha(), "dev_file": str(DEV.relative_to(REPO_DIR)),
            "dev_sha256": hashlib.sha256(DEV.read_bytes()).hexdigest()[:16],
            "index": retriever.version, "manifest": retriever.manifest,
            "dense_model": cfg.dense_model, "top_k": args.top_k, "prefetch_k": cfg.prefetch_k,
            "mode": cfg.mode, "bm25_weight": cfg.bm25_weight, "title_weight": cfg.title_weight,
            "omnidesk_k": cfg.omnidesk_k,
            "excerpt_coverage": EXCERPT_COVERAGE, "note": args.note,
            "latency_ms": {"p50": round(statistics.median(latencies), 1),
                           "p95": round(statistics.quantiles(latencies, n=20)[18], 1)}}
    out.write_text(json.dumps({"meta": meta, "metrics": metrics, "thresholds": table,
                               "suggested_min_score": best["threshold"], "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nОтчёт: {out}")


if __name__ == "__main__":
    asyncio.run(main())
