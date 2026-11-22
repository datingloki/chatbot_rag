"""Однократная генерация версионированного списка интентов S0 из FAQ.

    .venv/bin/python build_intents.py            # → intents/intents_v1.jsonl
    .venv/bin/python build_intents.py --version v2 --force

Одна строка = один интент (= одна статья FAQ):
  intent_id, title, answer_template, source_title, source_url, answer_key,
  examples  — примеры для сопоставления: заголовок, заголовок + начало ответа,
              перефразы FAQ из dev (с case_id и split_group_id, чтобы train.py мог
              исключать перефразы той же группы при подборе порога);
  status    — draft → approved (утверждает консультант 1С);
  source_sha256 — хэш статьи FAQ на момент генерации.

После генерации файл правится вручную (статусы, тексты шаблонов, примеры). Бот и train.py
читают интенты только из него; база знаний после генерации на S0 не влияет. Существующий
файл без --force не перезаписывается, чтобы не потерять ручные правки.
"""
from __future__ import annotations

import argparse
import json
from datetime import date

from data import FAQ_FILES, INTENTS_DIR, ROOT, _jsonl, load_dev, load_faq

ANSWER_HEAD = 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = INTENTS_DIR / f"intents_{args.version}.jsonl"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} уже существует (возможны ручные правки). Новая версия: --version, перезапись: --force")

    sha = {r["id"]: r.get("content_sha256") for p in FAQ_FILES for r in _jsonl(p)}
    src_file = {r["id"]: str(p.relative_to(ROOT)) for p in FAQ_FILES for r in _jsonl(p)}
    paraphrases: dict[str, list[dict]] = {}
    for c in load_dev():
        if c["faq_id"] and c["primary_class"] != "K1" and c["origin_type"] != "official_faq_entry":
            paraphrases.setdefault(c["faq_id"], []).append(
                {"text": c["message"], "kind": "dev_paraphrase", "case_id": c["case_id"], "group": c["group"]})

    INTENTS_DIR.mkdir(exist_ok=True)
    today = date.today().isoformat()
    n_par = 0
    with out.open("w", encoding="utf-8") as f:
        for faq in load_faq():
            iid = faq["intent_id"]
            examples = [
                {"text": faq["title"], "kind": "faq_title"},
                {"text": faq["title"] + "\n" + faq["answer"][:ANSWER_HEAD], "kind": "faq_title_answer"},
                *paraphrases.get(iid, []),
            ]
            n_par += len(paraphrases.get(iid, []))
            f.write(json.dumps({
                "intent_id": iid, "intents_version": args.version,
                "title": faq["title"], "answer_template": faq["answer"],
                "source_title": faq["source_title"], "source_url": faq["source_url"],
                "answer_key": faq["answer_key"], "examples": examples,
                "status": "draft", "approved_by": None, "approved_at": None,
                "source_faq_file": src_file[iid], "source_sha256": sha[iid], "generated_at": today,
            }, ensure_ascii=False) + "\n")
    print(f"{out.relative_to(ROOT)}: {len(load_faq())} интентов, {n_par} перефраз из dev")


if __name__ == "__main__":
    main()
