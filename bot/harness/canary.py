"""Canary-набор (design doc 7.3): 60 кейсов dev для быстрой проверки перед релизом.

    .venv/bin/python -m harness.canary build      # выбрать id и записать harness/canary_v1.json
    .venv/bin/python -m harness.canary check      # все id есть в текущем dev?
    .venv/bin/python -m harness.run --system rag --set canary --repeat 3

Правило выбора — по 10 кейсов на класс K1–K6, сначала рискованные, остальное — случайно с
фиксированным seed:
- K1: попытки вывести за контракт (инструкции «игнорируй/забудь», системный промпт), запросы
  профессиональной консультации (`prof_consultation`), посторонние вопросы с упоминанием 1С;
- K6: клиент-специфичные проблемы с признаками рабочей базы (доработка, «наша база», SQL, ошибка
  после обновления) — здесь бот не должен советовать править базу и обещать заявку;
- остальные классы — случайная выборка, разные `origin_type`.

Список замораживается файлом: при правке правила — новая версия (canary_v2.json), а не правка v1.
Набор — подмножество dev, поэтому числа canary не заменяют прогон dev и тем более eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from polybot.settings import BOT_DIR, REPO_DIR

DEV = REPO_DIR / "dev set" / "dev_300_balanced.jsonl"
CANARY = BOT_DIR / "harness" / "canary_v1.json"
PER_CLASS = 10
SEED = 20261120
CLASSES = ("K1", "K2", "K3", "K4", "K5", "K6")

INJECTION_RE = re.compile(r"игнорир|забудь|системн\w* промпт|инструкци\w* выше|ты теперь|притворись", re.I)
ONE_C_RE = re.compile(r"\b1\s?с\b|1c\b", re.I)
CLIENT_DB_RE = re.compile(r"доработ|наш\w* баз|рабоч\w* баз|sql|после обновлени|журнал регистрации", re.I)


def message(case: dict[str, Any]) -> str:
    return case["input"]["user_message"]


def risk_rank(case: dict[str, Any]) -> int:
    """0 — самый рискованный кейс класса, 3 — обычный."""
    text, tags = message(case), set(case.get("tags", []))
    if case["primary_class"] == "K1":
        if INJECTION_RE.search(text):
            return 0
        if "prof_consultation" in tags:
            return 1
        if ONE_C_RE.search(text):
            return 2
    if case["primary_class"] == "K6" and CLIENT_DB_RE.search(text):
        return 0
    return 3


def select(cases: list[dict[str, Any]], per_class: int = PER_CLASS, seed: int = SEED) -> list[str]:
    rng = random.Random(seed)
    ids: list[str] = []
    for cls in CLASSES:
        pool = [c for c in cases if c["primary_class"] == cls]
        rng.shuffle(pool)                                    # порядок внутри ранга — по seed
        pool.sort(key=risk_rank)                             # сортировка устойчивая
        ids += [c["case_id"] for c in pool[:per_class]]
    return ids


def load_dev(path: Path = DEV) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_ids(path: Path = CANARY) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))["case_ids"]


def canary_cases(dev: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    by_id = {c["case_id"]: c for c in dev}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"canary: в dev нет кейсов {missing[:5]}… — нужна новая версия canary")
    return [by_id[i] for i in ids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    args = ap.parse_args()
    dev = load_dev()
    if args.cmd == "build":
        if CANARY.exists():
            raise SystemExit(f"{CANARY.name} уже есть: canary заморожен, для нового правила — canary_v2")
        ids = select(dev)
        risky = sum(risk_rank(c) < 3 for c in canary_cases(dev, ids))
        CANARY.write_text(json.dumps({
            "version": "canary_v1", "rule": "harness/canary.py: по 10 на K1–K6, рискованные первыми",
            "seed": SEED, "dev_sha256": hashlib.sha256(DEV.read_bytes()).hexdigest()[:16],
            "risky": risky, "case_ids": ids}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{CANARY}: {len(ids)} кейсов, рискованных {risky}")
    else:
        cases = canary_cases(dev, load_ids())
        print(f"canary: {len(cases)} кейсов, все есть в dev")


if __name__ == "__main__":
    main()
