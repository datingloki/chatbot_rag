"""Загрузка FAQ (интенты S0) и кейсов dev/eval."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "knowledge base"
FAQ_FILES = [KB / "faq_articles.jsonl", KB / "faq_license_install_articles.jsonl"]
DEV = ROOT / "dev set" / "dev_300_balanced.jsonl"
EVAL = ROOT / "eval set" / "eval_300_balanced.jsonl"
INTENTS_DIR = Path(__file__).resolve().parent / "intents"
INTENTS_PATH = INTENTS_DIR / "intents_v1.jsonl"   # текущая утверждаемая версия интентов


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_intents(path: Path = INTENTS_PATH) -> tuple[list[dict], str]:
    """Интенты S0 из версионированного файла (build_intents.py) и sha256 этого файла."""
    if not path.exists():
        raise SystemExit(f"Нет файла интентов {path}. Сгенерируйте: .venv/bin/python build_intents.py")
    intents = _jsonl(path)
    ids = [i["intent_id"] for i in intents]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: повторяющиеся intent_id")
    return intents, file_sha256(path)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _as_obj(v):
    return json.loads(v) if isinstance(v, str) else v


def clean_text(text: str) -> str:
    text = text.replace(" ", " ").replace("?", "?")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_faq() -> list[dict]:
    """Все FAQ = все интенты S0. Статьи с одинаковым текстом ответа (дубли разделов
    Полиграфия 2 / Полиграфия 8) получают общий answer_key: выбор любой из них
    даёт тот же ответ."""
    faqs = []
    for path in FAQ_FILES:
        for r in _jsonl(path):
            sec = _as_obj(r.get("source_section")) or {}
            if isinstance(sec, str):
                sec = {"title": sec}
            text = clean_text(r["text"])
            faqs.append({
                "intent_id": r["id"],
                "title": clean_text(r["title"]).split("\n")[0],
                "answer": text,
                "source_url": r["source_url"],
                "source_title": sec.get("title", "") if isinstance(sec, dict) else "",
                "answer_key": hashlib.sha1(re.sub(r"\W+", "", text.lower()).encode()).hexdigest()[:12],
            })
    return faqs


def load_cases(path: Path) -> list[dict]:
    out = []
    for r in _jsonl(path):
        inp = _as_obj(r["input"])
        ctx = _as_obj(r["context"]) or {}
        out.append({
            "case_id": r["case_id"],
            "message": inp["user_message"],
            "history": inp.get("history", []),
            "primary_class": r["primary_class"],
            "expected_action": r["expected_action"],
            "faq_id": ctx.get("source_faq_id"),
            "origin_type": r.get("origin_type") or r["origin"].split(":")[0],
            "tags": _as_obj(r["tags"]) or [],
            "group": r.get("split_group_id") or ctx.get("group_id") or r["case_id"],
            "clarification_count": int(r.get("clarification_count") or 0),
            "clarification_limit": int(r.get("clarification_limit") or 5),
            "context": ctx,
            "raw": r,
        })
    return out


def load_dev() -> list[dict]:
    return load_cases(DEV)


def load_eval() -> list[dict]:
    return load_cases(EVAL)
