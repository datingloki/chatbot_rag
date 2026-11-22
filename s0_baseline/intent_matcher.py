"""Сопоставление обращения с интентами S0.

Интенты и их примеры берутся из версионированного файла (intents/intents_v*.jsonl,
build_intents.py). Примеры интента: заголовок FAQ, заголовок + начало ответа, перефразы.
Сходство обращения с интентом = максимум по его примерам от взвешенной суммы косинусов
нескольких представлений текста:
  char   — TF-IDF по символьным 2–5-граммам (опечатки, словоформы);
  lemma  — TF-IDF по леммам pymorphy3 (1–2-граммы);
  emb    — многоязычный sentence-эмбеддинг (ONNX через fastembed, открытые веса, CPU).

Векторизаторы и векторы примеров считаются один раз в train.py и сохраняются вместе с
моделью (pickle); при запуске бота кодируются только входящие сообщения. Сама ONNX-модель
в pickle не попадает и загружается лениво из s0_baseline/models/.

Уверенность интента — вероятность «топ-1 интент верный», откалиброванная логистической
регрессией на dev по признакам [сходство топ-1, отрыв от ближайшего интента с другим
ответом]. Порог по этой вероятности выбирается на dev (train.py), не на eval.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "domain_classifier"))
from domain_clf import lemmatize, normalize  # noqa: E402

MODELS_DIR = HERE / "models"


def _l2(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


class Embedder:
    def __init__(self, name: str):
        import warnings
        from fastembed import TextEmbedding
        self.name = name
        with warnings.catch_warnings():
            # fastembed >= 0.5.2 предупреждает о mean pooling для MiniLM. Это штатный пулинг
            # модели в sentence-transformers; порог и метрики S0 получены именно с ним.
            warnings.filterwarnings("ignore", message=".*mean pooling.*", category=UserWarning)
            try:
                self.model = TextEmbedding(name, cache_dir=str(MODELS_DIR), local_files_only=True)
            except Exception:  # первый запуск: скачать открытые веса ONNX в s0_baseline/models/
                self.model = TextEmbedding(name, cache_dir=str(MODELS_DIR))

    def __call__(self, texts: list[str]) -> np.ndarray:
        return _l2(np.array(list(self.model.embed(texts, batch_size=32)), dtype=np.float32))


class IntentMatcher:
    def __init__(self, intents: list[dict], weights: dict[str, float], emb_model: str | None = None):
        self.ids = [i["intent_id"] for i in intents]
        self.idx = {i: k for k, i in enumerate(self.ids)}
        self.answer_key = np.array([i["answer_key"] for i in intents])
        self.weights = {k: v for k, v in weights.items() if v}
        self.emb_model = emb_model
        self._emb = None
        corpus = [i["title"] for i in intents] + [i["answer_template"] for i in intents]
        self.char = TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(2, 5),
                                    sublinear_tf=True).fit(corpus)
        self.lemma = TfidfVectorizer(preprocessor=lemmatize, ngram_range=(1, 2), sublinear_tf=True).fit(corpus)

        protos = [(i["intent_id"], ex) for i in intents for ex in i["examples"]]
        self.proto_intent = np.array([self.idx[iid] for iid, _ in protos])
        self.proto_text = [ex["text"] for _, ex in protos]
        self.proto_kind = [ex["kind"] for _, ex in protos]
        self.proto_group = [ex.get("group") for _, ex in protos]
        self._proto_vecs = self._encode(self.proto_text)

    # ------------------------------------------------------------ pickle без ONNX-сессии
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_emb"] = None
        return state

    # ------------------------------------------------------------ encoding
    @property
    def emb(self) -> Embedder:
        if self._emb is None:
            self._emb = Embedder(self.emb_model)
        return self._emb

    def _encode(self, texts: list[str]) -> dict:
        out = {}
        if "char" in self.weights:
            out["char"] = self.char.transform(texts)
        if "lemma" in self.weights:
            out["lemma"] = self.lemma.transform(texts)
        if "emb" in self.weights:
            out["emb"] = self.emb(texts)
        return out

    def proto_similarity(self, texts: list[str]) -> np.ndarray:
        q = self._encode(texts)
        total = sum(self.weights.values())
        S = np.zeros((len(texts), len(self.proto_text)), dtype=np.float32)
        for name, w in self.weights.items():
            sim = q[name] @ self._proto_vecs[name].T
            S += (w / total) * (sim.toarray() if hasattr(sim, "toarray") else sim)
        return S

    def intent_scores(self, texts: list[str], mask: np.ndarray | None = None) -> np.ndarray:
        """cases × intents. mask (cases × examples, bool) исключает примеры (для CV)."""
        S = self.proto_similarity(texts)
        if mask is not None:
            S = np.where(mask, -1.0, S)
        out = np.full((len(texts), len(self.ids)), -1.0, dtype=np.float32)
        for j, i in enumerate(self.proto_intent):
            np.maximum(out[:, i], S[:, j], out=out[:, i])
        return out

    def top(self, scores: np.ndarray) -> list[dict]:
        """Топ-1 интент, его сходство и отрыв от лучшего интента с другим ответом."""
        res = []
        for row in scores:
            order = np.argsort(-row)
            best = order[0]
            other = next((j for j in order[1:] if self.answer_key[j] != self.answer_key[best]), None)
            margin = float(row[best] - (row[other] if other is not None else 0.0))
            res.append({"intent_id": self.ids[best], "score": float(row[best]), "margin": margin,
                        "runner_up": self.ids[other] if other is not None else None})
        return res


def conf_features(top: list[dict]) -> np.ndarray:
    return np.array([[t["score"], t["margin"]] for t in top], dtype=np.float32)
