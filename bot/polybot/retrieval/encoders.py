"""Кодировщики запросов и чанков — общие для индексации и поиска.

- Плотные векторы: модель fastembed (ONNX, CPU, открытые веса). По умолчанию
  `paraphrase-multilingual-MiniLM-L12-v2` — та же, что в S0.
- Разреженные векторы BM25 по леммам pymorphy3 (в S0 леммы заметно помогали). Индекс
  измерения — crc32 леммы, вес — tf-часть BM25; IDF считает Qdrant (`Modifier.IDF`).
"""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from functools import lru_cache
from pathlib import Path

TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.I)
STOPWORDS = frozenset(
    "и в во на не что как с со по к ко у о об от до из за для при же ли бы а но или то это "
    "этот эта эти тот та те так там тут где когда если чтобы который которая которые его ее её "
    "их им ими мы вы он она оно они я ты меня мне нас нам вас вам быть был была были есть "
    "можно нужно надо уже еще ещё все всё весь вся только также тоже очень а б".split()
)
BM25_K1 = 1.2
BM25_B = 0.75


@lru_cache(maxsize=1)
def _morph():
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=200_000)
def lemma(token: str) -> str:
    if token.isdigit() or not re.search("[а-яё]", token):
        return token
    return _morph().parse(token)[0].normal_form


def lemmas(text: str) -> list[str]:
    out = []
    for tok in TOKEN_RE.findall(text.lower()):
        lem = lemma(tok)
        if lem in STOPWORDS or (len(lem) < 2 and not lem.isdigit()):
            continue
        out.append(lem)
    return out


def term_index(term: str) -> int:
    return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF


class SparseBM25:
    """tf-часть BM25 для документа и единичные веса для запроса. Коллизии crc32 на словаре
    в десятки тысяч лемм редки и на ранжирование почти не влияют."""

    def __init__(self, avgdl: float):
        self.avgdl = avgdl or 1.0

    def doc_vector(self, text: str) -> tuple[list[int], list[float]]:
        terms = lemmas(text)
        tf = Counter(term_index(t) for t in terms)
        dl = len(terms)
        norm = BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avgdl)
        idx = sorted(tf)
        return idx, [round(tf[i] * (BM25_K1 + 1) / (tf[i] + norm), 6) for i in idx]

    @staticmethod
    def query_vector(text: str) -> tuple[list[int], list[float]]:
        idx = sorted({term_index(t) for t in lemmas(text)})
        return idx, [1.0] * len(idx)


def average_length(texts: list[str]) -> float:
    return sum(len(lemmas(t)) for t in texts) / max(len(texts), 1)


class DenseEncoder:
    """Обёртка fastembed. Для моделей e5 добавляются префиксы query:/passage:."""

    def __init__(self, model: str, cache_dir: Path, max_tokens: int | None = None):
        from fastembed import TextEmbedding

        self.model_name = model
        # Prefer the supplied cache: a network outage must not prevent loading existing weights.
        try:
            self._model = TextEmbedding(model_name=model, cache_dir=str(cache_dir), local_files_only=True)
        except (ValueError, OSError):
            self._model = TextEmbedding(model_name=model, cache_dir=str(cache_dir))
        if max_tokens:
            # fastembed берёт лимит из tokenizer_config модели (MiniLM — 128); можно поднять до
            # max_position_embeddings (512). Лимит должен совпадать при индексации и поиске.
            self._model.model.tokenizer.enable_truncation(max_length=max_tokens)
        self.max_tokens = self._model.model.tokenizer.truncation["max_length"]
        self._e5 = "e5" in model.lower()
        self.dim = len(next(iter(self._model.embed(["проверка"]))))

    def _norm(self, v) -> list[float]:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [float(x) / n for x in v]

    def passages(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if self._e5:
            texts = [f"passage: {t}" for t in texts]
        return [self._norm(v) for v in self._model.embed(texts, batch_size=batch_size)]

    def query(self, text: str) -> list[float]:
        if self._e5:
            text = f"query: {text}"
        return self._norm(next(iter(self._model.embed([text]))))
