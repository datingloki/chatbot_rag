"""Классификатор домена из `../domain_classifier` (обучен на dev, см. его README).

Решение = модель (`model/domain_clf.joblib`: пайплайн + пороги) + правила R1–R4 (`rules.py`).
Правила в joblib не хранятся, поэтому версия классификатора — хеш трёх файлов:
`domain_clf.joblib`, `domain_clf.py` (код признаков) и `rules.py`.

Модули загружаются по полному пути, без добавления папки в `sys.path`:
- `domain_clf.py` регистрируется под именем `domain_clf` — это имя записано в joblib
  (классы признаков), и под ним его импортирует `rules.py`;
- `rules.py` регистрируется под уникальным именем, чтобы не столкнуться с другим `rules`.

`predict()` возвращает `{"label": "in_domain"|"out_of_domain"|"unclear", "p_out_of_domain",
"used_history", "model_label", "rule"}`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

logger = logging.getLogger(__name__)

CLF_MODULE = "domain_clf"
RULES_MODULE = "polybot_domain_rules"


class DomainService(Protocol):
    version: str

    def predict(self, text: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path) -> ModuleType:
    """Импорт файла по пути под заданным именем. Если модуль с таким именем уже загружен
    из другого файла — ошибка, а не тихая подмена."""
    path = path.resolve()
    existing = sys.modules.get(name)
    if existing is not None:
        loaded_from = Path(getattr(existing, "__file__", "") or "").resolve()
        if loaded_from != path:
            raise ImportError(f"модуль {name!r} уже загружен из {loaded_from}, а нужен {path}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"не удалось загрузить {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def check_dev(meta: dict[str, Any], repo_dir: Path) -> dict[str, Any]:
    """Обучена ли модель на текущей версии dev.

    Если `train.py` записал `dev_sha256` — сравниваем хеш файла. В моделях, обученных до этого,
    хеша нет — тогда сверяем число обучающих кейсов и посторонних (как их считает
    `train.load_dev`: без кейсов с тегом `domain_unclear`, посторонние = K1).
    """
    dev_path = repo_dir / meta.get("trained_on", "")
    if not dev_path.is_file():
        return {"status": "dev_not_found", "dev": str(dev_path)}
    if meta.get("dev_sha256"):
        same = _sha256(dev_path) == meta["dev_sha256"]
        return {"status": "ok" if same else "mismatch", "by": "sha256"}
    rows = [json.loads(line) for line in dev_path.open(encoding="utf-8") if line.strip()]
    train = [r for r in rows if "domain_unclear" not in r.get("tags", [])]
    n, n_out = len(train), sum(r.get("primary_class") == "K1" for r in train)
    same = n == meta.get("n") and n_out == meta.get("n_out")
    return {"status": "ok" if same else "mismatch", "by": "counts",
            "dev_n": n, "dev_n_out": n_out, "model_n": meta.get("n"), "model_n_out": meta.get("n_out")}


class DomainClassifierService:
    def __init__(self, classifier_dir: Path):
        classifier_dir = classifier_dir.resolve()
        model_path = classifier_dir / "model" / "domain_clf.joblib"
        code_path = classifier_dir / "domain_clf.py"
        rules_path = classifier_dir / "rules.py"

        clf_module = load_module(CLF_MODULE, code_path)
        self._rules = load_module(RULES_MODULE, rules_path)
        self._clf = clf_module.DomainClassifier.load(model_path)

        shas = {"model": _sha256(model_path), "code": _sha256(code_path),
                "rules": _sha256(rules_path)}
        combined = hashlib.sha256("".join(shas[k] for k in sorted(shas)).encode()).hexdigest()
        self.version = f"domain_clf@{combined[:12]}"
        meta = self._clf.meta if isinstance(self._clf.meta, dict) else {}
        self.dev_check = check_dev(meta, classifier_dir.parent)
        self.info = {"version": self.version, **{f"{k}_sha256": v[:12] for k, v in shas.items()},
                     "meta": meta, "thresholds": {"t_in": self._clf.t_in, "t_out": self._clf.t_out},
                     "dev_check": self.dev_check}
        if self.dev_check["status"] != "ok":
            logger.warning("классификатор домена %s: модель обучена не на текущем dev (%s) — "
                           "перезапустите domain_classifier/train.py", self.version, self.dev_check)

    def predict(self, text: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        # Модель без правил, затем правила из файла, загруженного по пути, —
        # то же самое, что DomainClassifier.predict(use_rules=True).
        result = self._clf.predict(text, history=history, use_rules=False)
        label, rule = self._rules.apply_rules(text, result["model_label"])
        return {**result, "label": label, "rule": rule}


class FixedDomain:
    """Заглушка для тестов: метка по функции от (текст, история)."""

    version = "fixed-test"

    def __init__(self, rule):
        self.rule = rule
        self.calls: list[tuple[str, list[dict[str, str]] | None]] = []

    def predict(self, text: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        self.calls.append((text, history))
        label = self.rule(text, history or [])
        return {"label": label, "p_out_of_domain": None, "used_history": bool(history),
                "model_label": label, "rule": None}
