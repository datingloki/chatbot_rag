"""Классификатор выхода из домена чат-бота 1С:Полиграфии.

Признаки: TF-IDF по леммам (1–2-граммы) + TF-IDF по символьным n-граммам
+ усреднённые эмбеддинги navec (открытые веса, без внешних API — C12).
Модель: логистическая регрессия с балансировкой классов.

Выход — одно из трёх состояний маршрутизации (дизайн-док, 2.2):
    in_domain / out_of_domain / unclear
Порог unclear нужен для C05: сомнительный запрос не объявляется вне домена
без уточнения.

Использование:
    .venv/bin/python domain_clf.py "Посоветуй рецепт пирога"
"""
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model" / "domain_clf.joblib"
TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.I)


# ---------------------------------------------------------------- preprocessing
@lru_cache(maxsize=1)
def _morph():
    import pymorphy3
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=200_000)
def _lemma(tok: str) -> str:
    return _morph().parse(tok)[0].normal_form


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"1\s*с", "1с", text)          # «1 С», «1С» -> «1с»
    return text


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize(text))


def lemmatize(text: str) -> str:
    return " ".join(_lemma(t) for t in tokens(text))


@lru_cache(maxsize=1)
def _navec():
    import natasha
    from navec import Navec
    path = Path(natasha.__file__).parent / "data" / "emb" / "navec_news_v1_1B_250K_300d_100q.tar"
    return Navec.load(path)


class NavecMean(BaseEstimator, TransformerMixin):
    """Среднее navec-векторов токенов (по форме, иначе по лемме), L2-нормированное."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nv = _navec()
        out = np.zeros((len(X), 300), dtype=np.float32)
        for i, text in enumerate(X):
            vecs = []
            for t in tokens(text):
                for cand in (t, _lemma(t)):
                    if cand in nv:
                        vecs.append(nv[cand])
                        break
            if vecs:
                v = np.mean(vecs, axis=0)
                out[i] = v / (np.linalg.norm(v) + 1e-9)
        return sparse.csr_matrix(out)


class Lemmatizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [lemmatize(x) for x in X]


def build_pipeline(features: str = "lemma+char+navec", C: float = 4.0) -> Pipeline:
    parts = []
    if "lemma" in features:
        parts.append(("lemma", Pipeline([
            ("lem", Lemmatizer()),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ])))
    if "char" in features:
        parts.append(("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb",
                                              ngram_range=(2, 5), min_df=2, sublinear_tf=True)))
    if "navec" in features:
        parts.append(("navec", NavecMean()))
    return Pipeline([
        ("features", FeatureUnion(parts)),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000)),
    ])


# ---------------------------------------------------------------- data
def case_texts(case: dict) -> tuple[str, str]:
    """(текущее сообщение, сообщения пользователя из истории + текущее)."""
    inp = case["input"]
    if isinstance(inp, str):
        inp = json.loads(inp)
    msg = inp["user_message"]
    prev = [h["text"] for h in inp.get("history", []) if h.get("role") == "user"]
    return msg, "\n".join(prev + [msg])


# ---------------------------------------------------------------- inference
class DomainClassifier:
    def __init__(self, bundle: dict):
        self.pipe = bundle["pipeline"]
        self.t_out = bundle["t_out"]     # p_out >= t_out  -> out_of_domain
        self.t_in = bundle["t_in"]       # p_out <  t_in   -> in_domain
        self.meta = bundle.get("meta", {})

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "DomainClassifier":
        import joblib
        return cls(joblib.load(path))

    def p_out(self, texts: list[str]) -> np.ndarray:
        return self.pipe.predict_proba(texts)[:, 1]

    def route(self, p: float) -> str:
        if p >= self.t_out:
            return "out_of_domain"
        if p < self.t_in:
            return "in_domain"
        return "unclear"

    def predict(self, user_message: str, history: list[dict] | None = None, use_rules: bool = True) -> dict:
        """Решение по текущему сообщению; при неясности и наличии истории —
        повторная оценка с предыдущими сообщениями пользователя (короткие ответы
        на уточнение вида «да, в заказе»). Явный уход в постороннюю тему внутри
        диалога определяется по самому сообщению. Затем — слой правил (rules.py)."""
        p = float(self.p_out([user_message])[0])
        used_history = False
        if history and self.route(p) == "unclear":
            _, ctx = case_texts({"input": {"user_message": user_message, "history": history}})
            p = float(self.p_out([ctx])[0])
            used_history = True
        label, rule = self.route(p), None
        if use_rules:
            from rules import apply_rules
            label, rule = apply_rules(user_message, label)
        return {"label": label, "p_out_of_domain": round(p, 4), "used_history": used_history,
                "model_label": self.route(p), "rule": rule}


if __name__ == "__main__":
    clf = DomainClassifier.load()
    for text in sys.argv[1:] or [line.strip() for line in sys.stdin if line.strip()]:
        print(json.dumps({"text": text, **clf.predict(text)}, ensure_ascii=False))
